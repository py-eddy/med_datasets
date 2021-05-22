CDF       
      obs    F   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�9XbM�       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P���       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��9X   max       =       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�G�z�   max       @Fg�z�H     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�������   max       @vqG�z�     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @P@           �  6�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ˊ        max       @���           7`   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �e`B   max       >%�T       8x   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�g   max       B2�       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��2   max       B1��       :�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?���   max       C�]a       ;�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�&�   max       C�[       <�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          O       =�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          5       ?   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3       @    
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       Pc�7       A8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��*�0�   max       ?�e+��a       BP   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��1   max       =       Ch   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�ffffg   max       @Fg�z�H     
�  D�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�������   max       @vp(�\     
�  Op   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @!         max       @P@           �  Z`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ˊ        max       @��@           Z�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         =�   max         =�       \   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�n��O�<   max       ?�e+��a     �  ]                     0                              
      <      O   
         $      "   
   
                                                            '         )               	   I            
         .      *   "      -N`P�O�N�]�Ob>Ns��OpH�O��<N+9�O�� O$�NP��N'aO�/�N7�N�7�N�N��
PN���P���N��PO0� N�:�O��Oz^�O�wN�NMN�IN�<P
�N�S�N��N2�%N\��Oȹ2NB#�OO��OőN�n�N��*O�Ne�|N	iN�[�O��Nɱ|Ov+�N�AO+�O�2CPG�rN�jNjImNL�N���O�V�O#��O�N���N��M��N��O�Od-�Os�iO�}N��?N�d���9X�e`B���
�D����o��o��o$�  $�  ;o;D��;��
;ě�<t�<t�<49X<49X<49X<D��<D��<D��<e`B<e`B<e`B<u<u<u<u<u<u<u<u<�o<�o<�C�<�t�<�t�<�t�<�t�<�t�<���<���<��
<��
<��
<��
<��
<��
<�1<�9X<�9X<ě�<ě�<ě�<���<�`B=+=��=0 �=49X=8Q�=P�`=T��=Y�=e`B=�o=���=���=��=��������������������?=>BEJMN[eglmhg[ZNB?;BFNS[fghga[XNNB;;;;mmpoxz����������zwqmDBINP[cf`[TNDDDDDDDDvx�����������������v� 
#/<GHGC</#
 �����

�����������'/HUanz������na\UH<'!")5BNY[__][QNB<5*)!`anz����zrna````````5BN_gtz}|g[NB5)!#0990)#!!!!!!!!!!�����)-102-��������������������������������������=9:BJOR[\]_bc[OGCB==�������������
)N[r���tg[P5
������������������������/<npz}zqb/#�����~�����������������jfnz������������zonjIO\hu{�uh\WPOIIIIII���������������������������������������)5BLQU_YOB95)135?BN[`ggig^[NB9511geegt�����tkggggggg������������������������������
#)$���).6@BIKKB6,)��������������������{|��������{{{{{{{{{{fbfggty����tlgffffff
/<UdmnfaUHB/PSUacckga^VUPPPPPPPP23BOQ[ghstxtoh[OB:62��������� ����������������������������ORV[ht}~vth[OOOOOOOO��������������������"/;=694//$"?@HTakfaTH??????????�������������������������������������������������������������������������������#/<GMTV\^VH</#��������������������#06<IU]`XUKI@<0%%1BO[gjjmlg[N5)5/@[d�����������[NB5mnoprvz��������zunmm�������������������������������������������������������������������&+/.)��&" !&)-5BINOPXNB5/)&��������������������)55753)����������������������������������������DFHHPTaeba^VTHDDDDDD��������������������������
������������
&)15)���7;=BKOX[hlpolhe[XOB7���������������������������������������̺������������������������������������������
���#�/�<�?�>�/�-�#��
�	����������Ź����������ŹŶŭŧŠŖŠŭŭŴŹŹŹŹ����������������������������������������čĚĦĨĦĠĚčā�|āĄčččččččččĚĳĽ��ĿĻĵĳĦĚčćā�y�|�~�}āč�y���������������y�m�`�T�J�A�;�<�I�`�m�y�"�.�9�4�.�"�����"�"�"�"�"�"�"�"�"�"�	����������۾ʾ��������ʾӾ޾�	�������������������z�m�h�a�^�f�m�s�z�{���<�?�C�A�>�<�9�/�,�/�2�:�<�<�<�<�<�<�<�<�\�uƟƪƨƚƕƇ�u�h�\�<�8�:�B�E�D�H�O�\�ּ��������ټּԼּּּּּּּּּ��U�b�{ŇŔŝŠŠőŇ�n�U�I�<�3�7�<�>�I�U�;�G�T�`�f�`�\�T�G�;�6�7�;�;�;�;�;�;�;�;�n�zÇÏÓ×ÕÓÇ�{�z�z�n�c�d�h�n�n�n�n�����ʾҾ׾ؾ׾ξʾ�������������������������������������������������������������������)�5�:�G�H�B�:�5�������������꿒���������ĿƿѿĿ��������������������������A�M�^�[�A�5�(������Կҿؿؿѿ�����������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�EͿ� �!����	���������	�
��������(�5�?�E�H�D�A�/�(����������	������!�6�@�C�G�C�6�#���������������;�T�`�m�w�����y�m�`�T�;�.�"��	������;������!���������������������)�6�B�E�H�B�A�6�)�����'�)�)�)�)�)�)�������������������������������������������������)�B�[�b�n�p�[�N�5�������������żʼԼּټּּʼ���������������������ìù������������ùìàÝÜàåèìììì���	���	�������������������������������#�0�<�>�<�4�0�*�#������#�#�#�#�#�#��"�/�;�5�0�0�-�"�	�����������������	�������������������������������������������#�)�/�/�5�/�/�#�!���
�� ���
���M�f�v�|�}�{�{�����Z�M���������8�MÓàìîùþýùìëàÓÇÁ�|�}ÇÉÓÓ�@�L�Y�\�c�d�Y�L�@�>�7�8�@�@�@�@�@�@�@�@�O�[�h�tāăĊā�t�h�[�O�M�N�O�O�O�O�O�O�"�/�;�H�T�Z�a�b�a�c�a�T�H�;�1�/�"�!�!�"�;�H�I�S�P�H�;�8�5�8�;�;�;�;�;�;�;�;�;�;�����������������������#�/�<�H�M�R�H�F�=�<�/�'�$�#���#�#�#�#��"�;�G�T�b�g�_�T�G�"��	���������	��m�v�y����|�y�m�`�\�T�K�H�T�`�d�m�m�m�m����!�$�)�)����������������������T�a�g�m�u�m�a�T�O�S�T�T�T�T�T�T�T�T�T�T���������������������}�r�p�f�d�_�f�r��ݿ������%�(�1�2�(����ٿ˿Ŀ����ݾ�(�A�L�Z�n�������f�Z�(������	�����������������������������������������������������������������������������������ŇŔŠŦŠŘŔŇŁŃŇŇŇŇŇŇŇŇŇŇ���ûлܻ����������ܻл˻û������������	��"�/�;�H�O�T�U�R�H�;�"�	�����������	�����������������������������~���������������� �����������ܻٻػܻ���¦²¶¿¿¿½²±¦¦¦¦¦ÓàçâàÖÓÑÏÐÓÓÓÓÓÓÓÓÓÓ�l�x�������������x�u�l�k�l�l�l�l�l�l�l�l�b�n�{�{ŇŉŏŇ�{�v�n�b�_�]�b�b�b�b�b�b���ʼ�����������ּʼ���������������"�/�;�D�F�A�;�/��	�������������	���:�F�S�_�f�f�]�S�F�-�&�!�������!�:��!�-�:�F�J�S�U�X�S�F�:�-�$�!������Y�e�l�r�~�����������~�x�r�e�c�Y�V�O�Y�YD{D�D�D�D�D�D�D�D�D{DxDoDiDiDoDpD{D{D{D{ H = ] D : /  8 _ E } + 3  h  L : X E S 4 , T ! ' J 5 3 < k B I D Z U n O F $ J N F Y \ D & 1 ! ; T 2 ` O r R � % R / = p R ] Q > V M i 1  �  E  �  �  {  �  U  ?  �  o  �  H  B  �  K  �    �  �    �  �  y    �  �    "  �  �     �    N  w    m  f  5  9  �  �  G  �  @    �  �  �  ;  \  #  �    �  $  �    �     �  <    �  s  �  A  b  �  �e`B:�o$�  <�9X;��
<�9X=@�;��
<�1<�t�;�`B=t�<49X=t�<49X<�9X<�1<�t�=�t�<���=�Q�<ě�=t�<���=L��<��=D��<���<���<�h=�w<���<�<��
<�j=@�<�9X<���=D��=8Q�=C�=+<���<ě�<�9X=o=�w<���=u<�`B=��=�o=P�`=\)<�=+='�=�G�=}�=�hs=m�h=y�#=aG�=u=��=��w=���=�l�=�>%�TB"VB�Ba�B ?�B�BgcB�B��B`�B�yB��B�B%jB��BmVB�LB�BGBb�B�mB�?B �5B��B2�B�]B�,B�B2�B	��B�SB"�B��B�`B
��B	�;B�fB1hB�B"+�B"%gB��B�A�gA�]�B�B�qB!�B-nB��B ��B&0�B��B
��Bz3B*u@BzQB�nBSBb�B�OBTwB�KB,��A�a�B]EBv�B�VB*�B~�BK�B"?�BhB�'B <5BQ�Bz�B�BG�B��B<{B�BIB%��B��B��BǱB��BbpB?�B�dB;B �gB?�B1��B�6B��BB�B�B	�.BñB�9B�)B@*B
�3B	��B��B?BB"��B"@�BêB@	A��2A�V{B�.B�B!X�B-0B�9B ��B&?�B#�B%B��B*GxB@�B9�BHBA�B��B��B@tB,��A�{BBB@8B��B?�B�1B�@ =UA���A���A�9A���A߉]Ak�A_��AU��A�'�Aû>BX
A�}A�}EAe�wAȬBAM��Ar��A��?AtI�A�"�A�C�]aA[��A�|A��Ac��A���A���A��AA�8�@���A��A��/A���A��fA�I�A��A:�A�Y�?���A۝�A��6A�8A�,�A��oA_�Ajd�A�'�A�n|@���A�c�A<��A��lA!ZA�ih@��A��2A���@���A��OA���@�� A���A ��A�sV@z��@yӫ?��C��m@��A�~A��6A��A�UTAߣ=Ak TA_u�AU�A���AÔ�B]4AA�.Ae	RAȂ�AM lAs,�A��At��A��A���C�[A[�A�{�A�A$Ae�|A�n�A��A�FA��@��]A�myA�~]A�lA���A�r�A��cA7�Aˀ�?�&�AہA�.�A�x3A�\�A���A_�Aj�)AҐ�A���@���A�pAA< �A�wuAϝA�#@��A���A��
@��A���A�~�@�A���A �dA�p�@s�P@{�
@2&C���   	                  1                                     =      O            $      "                                                                  (         )               	   I                     .      +   #      .                           '         %                     '      5                  %            -               !         %                        #               !   3               #                                                               %                                    3                  %            +                        !                                          !                                                   N;��N�rqN�]�O�Ns��O#�OJ�N+9�O�@N���NO��FN'aOwN7�N�&N�N��
Ou��N��GPc�7N�ΦNˏ�N�:�O �DO%�O�wN��N��N���O�N�S�N��N2�%N\��O��&NB#�Nd=�O��;N��oN�n�N��*N��Ne�|N	iN���O�p�Nɱ|Oa0N�AN3)%O���O���N�jN"�hNL�N���O�Z�O�N���N���N��M��N��Oj�jOd-�OS��N�i�N��?N�d�  �  L  �  �  �  �  �  �  w  I  �  h  !  z  �  �  �  z  }    �  h  �  �  C  �  @  T  �  k  �  �  d  �  �  �  P  �    �  o  �  �  R  h  �  ^  �  �  �  �  �  o  �    �  �  
T  l  �  F    �  ?  	+  �  m  d  �  ���1�49X���
;D����o;ě�<T��$�  :�o;�`B;D��<D��;ě�<�C�<t�<D��<49X<49X=+<e`B<���<u<�1<e`B<ě�<��
<u<�o<�C�<���<�t�<u<�o<�o<�C�<ě�<�t�<�1<�j<���<���<���<�1<��
<��
<�j<���<��
=o<�9X<�`B=t�=o<ě�<���<�`B=+=u=8Q�=@�=8Q�=P�`=T��=Y�=��=�o=��=�9X=��=��������������������A@BHLN[agjjgd[QNJBAA;BFNS[fghga[XNNB;;;;uqqtvz����������}zuuDBINP[cf`[TNDDDDDDDD��������������������	
#/7<?CA</#
���

�����������*/HUanz������na]UH<*+.5BNQZ[\[WNFB85++++`anz����zrna````````)5BNgpvwung[N5)&!#0990)#!!!!!!!!!!������()*+)$������������������������������������=9:BJOR[\]_bc[OGCB==�������������)BNV[a[ZKB5)������������������������/<M]qxvl]/#�����������������������xuxz�����������zxxxxIO\hu{�uh\WPOIIIIII����������������������������������������)5BLQU_YOB95)255ABN[]eggg][NB=522oigit{�����toooooooo����������������������������
 '& �����).6@BIKKB6,)��������������������{|��������{{{{{{{{{{fbfggty����tlgffffff#/<HU]ahhaXUH<PSUacckga^VUPPPPPPPPDEO[ehnhf[SODDDDDDDD����������������������������������������ORV[ht}~vth[OOOOOOOO��������������������""#/3//.-#" ?@HTakfaTH??????????�������������������������������������������������������������������������������! #/<HLRULH</%#!!!!��������������������/'06<IILIB<0////////!-5BN[fhhb[NB50)']`fs��������������j]mnoprvz��������zunmm������������������������������������������������������������������"%&%!��(#!#)5?BGMMNB51)((((��������������������)55753)����������������������������������������DFHHPTaeba^VTHDDDDDD��������������������������
�����������$)(������NEGOP[hkkih^[ONNNNNN���������������������������������������̺����������������������������������������
���#�/�7�8�/�)�#���
���������
�
Ź����������ŹŶŭŧŠŖŠŭŭŴŹŹŹŹ����������������������������������������čĚĦĨĦĠĚčā�|āĄčččččččččĚĦĳĵĹĵĳĪĦĚčćăĄąčččč�`�m�y��������������y�m�`�U�N�I�L�T�Z�`�"�.�9�4�.�"�����"�"�"�"�"�"�"�"�"�"�	����������վʾ��������ʾվ߾�	�����������������z�u�m�g�m�q�z�}���������<�?�C�A�>�<�9�/�,�/�2�:�<�<�<�<�<�<�<�<�\�h�uƃƍƙƚƍƁ�u�h�\�O�G�@�C�K�M�O�\�ּ��������ټּԼּּּּּּּּּ��U�b�n�{ŇōŔřŕŇŃ�{�n�b�U�I�G�E�I�U�;�G�T�`�f�`�\�T�G�;�6�7�;�;�;�;�;�;�;�;�n�zÆÇÓÕÔÓÇ�z�n�e�f�l�n�n�n�n�n�n�����ʾҾ׾ؾ׾ξʾ�������������������������������������������������������������������'�)�0�5�7�5�3�)������������������������ÿ�������������������������������(�>�J�[�W�N�A�(������ٿؿ޿߿������������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��� �!����	���������	�
��������(�5�7�?�A�C�A�;�5�(���
��������*�6�8�?�:�6�/�*����� ��������;�T�`�m�w�����y�m�`�T�;�.�"��	������;������ ����������������������)�6�B�F�B�>�6�)�����������������������������������������������������������B�[�k�l�[�N�5�)�����������Ѽ��żʼԼּټּּʼ���������������������ìù������������ùìàÝÜàåèìììì���	���	�������������������������������#�0�<�>�<�4�0�*�#������#�#�#�#�#�#�	��"�,�.�,�,�*�'�"��	���������������	������������������������������������������#�'�*�#����
��
����������M�Z�f�m�t�v�l�f�M�A�(�������(�A�MÓàìùùùùìäàÓÉÇÃÇÇÓÓÓÓ�@�L�Y�\�c�d�Y�L�@�>�7�8�@�@�@�@�@�@�@�@�O�[�h�tāăĊā�t�h�[�O�M�N�O�O�O�O�O�O�/�;�H�T�W�a�a�a�\�T�H�H�;�/�"�"�"�%�/�/�;�H�I�S�P�H�;�8�5�8�;�;�;�;�;�;�;�;�;�;�����������������������/�<�H�H�L�H�C�<�9�/�*�'�#�#�#�&�/�/�/�/��"�.�;�I�T�V�Y�T�N�G�.��	��������	��m�v�y����|�y�m�`�\�T�K�H�T�`�d�m�m�m�m���������������������������������T�a�g�m�u�m�a�T�O�S�T�T�T�T�T�T�T�T�T�T�r�������������{�r�l�p�r�r�r�r�r�r�r�r������������ݿԿο˿̿ѿڿݿ���4�A�M�Z�f�{�~�z�s�f�Z�M�4�(�����(�4��������������������������������������������������������������������������������ŇŔŠŦŠŘŔŇŁŃŇŇŇŇŇŇŇŇŇŇ���ûлܻ����������ܻл˻û������������	��"�/�;�F�L�K�E�;�/�"��	�������� �	���������������������������������������������������������ܻۻۻܻ���¦²¶¿¿¿½²±¦¦¦¦¦ÓàçâàÖÓÑÏÐÓÓÓÓÓÓÓÓÓÓ�l�x�������������x�u�l�k�l�l�l�l�l�l�l�l�b�n�{�{ŇŉŏŇ�{�v�n�b�_�]�b�b�b�b�b�b�ʼ˼ּ�������ּ˼�����������������"�/�;�D�F�A�;�/��	�������������	���F�S�_�a�c�[�S�F�-�!���	�
���!�-�:�F�!�-�:�E�F�P�N�F�:�-�+�!�� �!�!�!�!�!�!�Y�e�l�r�~�����������~�x�r�e�c�Y�V�O�Y�YD{D�D�D�D�D�D�D�D�D{DxDoDiDiDoDpD{D{D{D{ B . ] A : .  8 _ J } ( 3  h ! L : D < K . 6 T  . J 7 2 A m B I D Z N n I L , J N T Y \ M # 1  ; C  3 O t R �  F 1 = p R ] L > V / i 1  _  �  �  Y  {  N  �  ?  �  �  �  �  B  �  K  �    �    �    �  �    W  M        �  �  �    N  w  I  m  �  �  �  �  �  �  �  @  �  
  �  4  ;  O  %  �    �  $  �      �  �  <    �    �  �  �  �    =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  �  �  �  �  �  �  �  �  q  b  R  ?  +    �  �  �  �  �  �  @  H  J  K  L  K  B  4  "    �  �  �  r  H    �  �  f    �  �  �  �  �  �  �  �  �    n  \  J  7  $    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  H    �  ^  �  2  Z  e  �  �  �  �  �  �  �  �  �  �  �  �  �  |  o  c  H  �  �  1  S  r  �  �  �  �  �  �  �  �  �  �  b  7     �  c  �  Z  �  �  �  <  w  �  �  �  �  �  f  9  �  �    �  $  �  �  e   �  �  �  �  �  �  �  �  �  �  v  i  [  H  1      �  �  y  I  r  w  w  v  q  f  W  I  6    E  @  )    �  �  �  �  `  �  �  �    &  7  C  H  D  6  #  	  �  �  |  =  �  �  �  f    �  �  �  �  �  �  �  y  n  b  U  E  6  &    �  �  �  ]  -  �  %  D  Z  f  g  ^  O  7    �  �  �  \    �  �  >  �  T  !  $  &  )  +  +  +  +  *  (  &  $            �  �  �    (  J  `  o  v  y  k  E    �  �  �  �  �  e    �  L  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  h  X  D  -    �  �  �  f  ,  �  �     �  �  �  �  �  �  �  �  �  �  �  o  P  $  �  �  j  )   �   �   Z  z  w  t  o  i  b  X  M  B  3  $      �  �  �  �  �  {  W  �  C  }  �    U  t  |  i  U  <    �  o  �  M  �  �  5  Q  �          
  �  �  �  �  �  V  +    �  �  �  T  !  �  �  �  �  �  �  �  �  �  �  �  |  b  +  �  \  �  U    �  �  ^  c  g  `  W  J  <  ,      �  �  �  �  �  �  �  �  �  �  L  w  �  �  �  �  �  �  �  y  ]  ;    �  �  >  �  V  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  b  K  .     �   �  �  �    6  A  C  @  9  +      �  �  [    �  <  �  �  �  `  s  �  �  �  �  �  �  �  �  �  c  >    �  �  }  Q  #  �  @  7  %            �  �  �  �  �  h  A    �  q  �  N  �  J  O  T  S  P  G  9  "    �  �  �  �  ]  ;      �  �  �  �  �  �  �  �  �  �  �  �  {  h  S  <  %    �  �  �  �  T  T  \  b  f  i  k  j  f  ]  U  L  A  3  $      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ^  /      �  �  @  �  �  �  �  �  �  �  �  �  t  f  Y  N  B  8  /  $  	  �  �  #  d  T  B  /    �  �  �  �  �  �  �  �  ^  2    �  �  ?  �  �  �  �  x  m  a  X  O  E  <  2  '        �  �  �  �  �  �  �  �  �  �  �  �  x  j  \  H  -    �  �  �  �  �  d  E  m  t  v  �  �  �  t  h  X  A  $  �  �  �  L    �  o  �    P  H  A  :  2  )    
  �  �  �  �  �  �  �  {  f  Q  <  '  r  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  e  @    �            �  �  �  �  �  �  �  �  t  ,  �  m  �  R  �  �  �  �  �  �  �  �  z  W    �  w    �  9  �  -  T   e  o  Z  B  *    �  �  �  �  z  Y  8    �  �  \  �  e  �  g  �  �  t  R  -    �  �  �  i  J  5  $  (  :  X  �  �  �  L  �  �  �  �  �  �  s  _  I  3      �  �  �  �  y  x  w  v  R  M  I  D  @  9  .  "      �  �  �  �  �  �  �  ~  j  U  h  l  q  v  z    �  �  �  �  �  u  a  M  8  $    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  g  P  4    �  �  k  H  O  V  Z  ]  ^  ^  Y  Q  G  ;  ,        �  �  �  v  =  �  �  �  �  �  �  �  �  �  �  �  �  �  p  `  O  4     �   �  �  #  U  }  �  �  �  �  t  N  $  �  �  �  ;  �  v    �  Q  �  z  o  e  Y  L  @  1  !      �  �  �  �  �  �  �  t  _  �  j  ]  [  w  �  �  �  �  �  �  �  a     �  �  N     �   [    7  Y  s  �  �  �  {  U  '  �  �  r  #  �  O  �    W  s  5  3  %    ;  ]  n  h  [  M  B  9  -    �  �  �  Y    �  �  �  �  �  �  �  w  `  M  8  !    �  �  �  �  �  �  �  x  �                +  1  -  (  $            
  	    �  �  �  �  �  �  �  }  p  Z  D  .    �  �  �  �  |  \  <  �  �  ~  e  L  @  X  q  i  ^  Q  D  6  ,  "  <  a  r  a  O  	4  	�  	�  
  
A  
Q  
S  
M  
<  
  	�  	�  	T  �  Y  �  �  �  �  O  9  G  l  ^  L  7  !    �  �  �  x  L    �  �  	  �    =  �  �  �  �  �  �  �  |  W  *  �  �  ~  8  �    �  �  �  )  F  8  (       �  �  �  �  d  =    �  �  �  o  &  �    �      �  �  �  �  �  �  �  �  k  R  7    �  �  �  �  �  b  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ?  6  .  #      �  �  �  �  �  �  k  U  E  ;  2  5  ;  B  �  	  	  	)  	+  	$  	  �  �  {    �  +  �    �  �  D  %  �  �  �  �  �  c  G  #  �  �  �  {  S  =  &    �  �  �     �  -  j  e  c  S  ;    �  �  �  D  �  �  F  �  _  �  �  �  �  �    !  :  S  d  b  W  D  *  �  �  m    �  M  �  M  �  7  �  �  �  �  �  �  s  X  7    �  �  �  g  :    �  �  D  ^  �  e  ?    �  �  D  �  �    �    �  
  
�  	�  	a  �  :  �
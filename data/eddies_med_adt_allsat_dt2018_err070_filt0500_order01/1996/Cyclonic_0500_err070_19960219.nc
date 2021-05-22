CDF       
      obs    L   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?pbM���   max       ?�z�G�{     0  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N	�   max       P�E     0  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��1   max       <u     0      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�(�\   max       @F��Q�     �  !<   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��
=p��    max       @v{�z�H     �  -   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @Q            �  8�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @���         0  9�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���   max       �49X     0  :�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��g   max       B55�     0  ;�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��m   max       B50     0  =$   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =���   max       C��     0  >T   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >)��   max       C��+     0  ?�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          P     0  @�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?     0  A�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9     0  C   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N	�   max       P���     0  DD   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�$tS��N   max       ?ո���*     0  Et   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��1   max       ;�`B     0  F�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�ffffg   max       @F��Q�     �  G�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��
=p��    max       @v{�z�H     �  S�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @Q            �  _�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @���         0  `,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A�   max         A�     0  a\   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?|C,�zxm   max       ?ո���*     �  b�   #   /      
         
                     N   0   "         0            #   <   &                        (   ,   7         (   1   F               	   ,   E      
       E      
            (            :                   .            #   	   
   *   P   O�6pO��O��eN�UO��N*�vN���NN�Nx��O�3�N�8O��	Nd�VP�EO��O�L�N���N|GPS�O9%NTwBO,��P�3O� O�rOB&N.�O�`�Ofc�N`��O�=�P��O��O��Py�N�CO=nO��O��#O�S�O>�9N�R(N�0nOHyxN�B�Pr�P5,�N?\�N��O�TfO�`>N	�O3mN/��Nf&O&��OQVN���N�g{O;�P��Ok��Ne}�O�{N��FN���OO�O���N?�O�QOhbLN�FN"�ON��P�QN$K`<u;��
��o�o�o���
�ě��o�o�D���T���T���T���e`B�e`B�u�u��C���C���9X��j��j��j������/��/��/��/�����o�o�+�+�+�C��t��t��t��t���P��P�����#�
�#�
�'''',1�,1�,1�0 Ž49X�8Q�8Q�<j�@��@��@��H�9�H�9�P�`�P�`�Y��Y��Y��aG��ixսu�u��7L��������1������#)*)
�������#HUafkjgaUH</#<IUbgjb\UI<0-#���

������������ #*46ACGJLMKEC7*��������������������imz����zwrmlgeiiiiii56@BOUROGEEBB@635555>BCKOUYXUOOB@=>>>>>>������	"&���������������������������������������������NOU[ehkkha[XOONNNNNN!5Nk��������tB��������� ����������&5Ngsuoj_[NB5)##./422/#�����
#$#
������#0<In{�����{n>0# #!)36A?65,)��������������������|���������������}yx|������ ���������������������	����������
#/1;6/*#
�����LOT[hksqlmmh[WPOMILLdht���thcdddddddddd��������������������)5??BBDDB50+��������������������]amz���������zvws`]]��������������amz���������zma^ZY[a������

�����)6Ot�����}{th[XD1%$)����������������������������������������������������������������)08=@6)������AKIU\bfkllhaUI<3006AEHLUZaehhedaXTKHECCETUX`alnrnmnnnba]UTTTz������������}zzzzzz����������������������������������������">Ubn{�����{bUI?+%*"������� ����������459?BGKKGB5344444444��������������������������������IQN[dj������tg[LA<:I��������������������)6BLS[[SQOHB?763351)����

�����������fgt��������vtgffffff������  ��������������������� �����������������������������_aalmtz��~zuma______������������������������,15DD9������fgtw����������xthdcfx�����������~txxxxxx5B[grogXTNB5������


�����������������������������������������������29HUanz����znaUH<9/2��������������������UUanzz�zzrnaVUQNOUUCHUakqrrppsniaUHB?@Cvz����������zytvvvv����������������������
#&#!
�����
/;GISUP/#
�����������������������������������������������"�(�'�"��	����4�1�.�0�3�;�A�M�Z�f�m�q�y�t�u�o�Z�M�A�4�����������������ȼּ������ټʼ�����ŔœōŔŚŠŭŲŸŴŭŠŔŔŔŔŔŔŔŔ�"��	�����������	��"�%�.�-�0�.�"�#����#�/�;�2�/�*�#�#�#�#�#�#�#�#�#�#�6�/�-�6�:�B�K�J�O�[�_�[�O�B�6�6�6�6�6�6�S�Q�F�E�@�F�S�X�_�l�x�z�x�l�c�_�S�S�S�S�����������������������������������������y�m�`�V�L�L�T�`�p�y�����������������{�yÇÄ�z�o�zÇÓàìôõìàÓÇÇÇÇÇÇ���������������$�0�9�7�9�5�:�:�2�$��e�c�Y�M�Y�]�e�r�x�~���~�r�r�e�e�e�e�e�e�׾��������׾�����	���%�9�+�/�)���׽нɽĽ��������������нݽ������������=�;�0�+�*�$�0�=�V�b�o�r�t�r�o�g�b�V�I�=�H�F�<�1�<�>�H�U�a�e�j�a�U�T�H�H�H�H�H�H�����������������������������������������f�N�D�?�4�7�>�c�s���������������������f²§¦¦²¿¿����������¿²���������������������������������������������������Ŀѿݿ������������ݿѿĿ����y�p�w�����ѿ���(�8�?�5�(���ݿĿ������g�N�(�������5�A�Z�g�s�������������g�ҾʾǾ¾��Ⱦ־׾���������������ҹù����������ùϹܹ�����������ܹϹùú����$�'�(�1�.�'�������������|�k�d�a�m�y���������ĿǿпտӿĿ���������������������������� ������������������
����)�5�B�N�B�;�5�)��������s�i�T�G�H�g�s�������������������������s�Z�N�A�(���'�A�N�Z�g�s�y�����������s�ZưƤƝƚƖƏƚƳ����������������������ư���������ööù���������)�-�)��#� ��n�g�n�e�g�����������������������������nìæàààâìù����������ùìììììì�a�\�Z�^�a�m�u�z���������������}�z�m�a�a�����������������ûܻ���� �������ܻл���������������'�3�1�7�7�2�'�!�����н��½нսݽ����4�A�M�Q�P�A�4�(������������5�A�N�Z�`�g�g�Z�S�A�(���ѿϿĿ��������Ŀпѿտݿ޿����ݿѿѾ����������������ƾʾѾӾϾʾ������������f�Z�V�X�Z�b�f�s��������������������s�f�����������������������������������������л������������û��@�G�R�W�M�M�4������)�������B�[�tčğķĲĦč�t�[�C�)�������������������	������������������M�Q�Y�`�f�r���������������������r�Y�M�ʼ��Ƽռڼ������!�3�7�5�!�����ּ����������������*�6�S�_�e�c�\�O�C�*���
�������&������������m�d�n�y�����������������������������y�m�ɺƺ������ºɺֺ�ݺֺ̺ɺɺɺɺɺɺɺ�����ÿûü�������������������������������=�;�0�,�$�'�0�=�I�V�b�k�o�r�o�m�b�V�I�=�ּӼʼļ������üʼӼּ��������ּ'�"��&�'�4�@�M�V�Y�\�Y�R�M�@�4�'�'�'�'����������������������������������������ĿļĸĺĿ����������������������������Ŀ�y�D�0�.�S�`�y�Ľн��(�"����ӽ����y���������������������
���(�#�!�
������ŠŗŠţŠŝŠũŭŹ��źŹŭŠŠŠŠŠŠ�U�H�?�:�<�9�/�0�<�H�M�a�nÏÒÑ��n�a�U����������������������������������������������������������������������h�_�[�U�W�[�`�hāčĚĦıĳĦĚčā�t�h���z�o�g�h�k�p�y�z����������������������E�FFF$F1F5F3F1F$FFFE�E�E�E�E�E�E�E����������	���"�%�(�"�!��	����𹝹��������������ùϹܹ����ٹù���������������������������������������!������������D�D�D�D�D�D�D�D�D�D�EEEEEEED�D�D�EPE*EE E*E;ECEPE\EuE�E�E�E�E�E�E�E�EuEP���������������������������������������� % B H , I Y F f V E u V - S [ 7 7 w 0 + 2 5 p X . E 6   W � P 2 H Z H $ L $ O [ i M V 6 < 0 J a i X , g Z = n 9 h % * ! H 7 G J 8 \ B M p  @ S 1 @ ; D    �  M  �  X  e  �  �  �  �  �  �  f  ?  f  �  �  �  �    U  r  m  �    -  K  z  
  �  �  T  �  
  �  �  /  �  �  �    �  �  �  '     F  W  \  �  F    B  N  �  l    �  �  �  �  �  x    �  �  �  +  �  7  �  �  2    �  B���
�,1�ě��49X�T���D���u�T���e`B�o�ě������ͽ�^5�}�D����`B��1�����w�����#�
�ixս�{����D�����,1�e`B�C��D���m�h������㽲-�T���49X���㽮{��
=�]/�49X�0 ŽY��H�9�����G��D���P�`�����S��<j�P�`�L�ͽL�ͽ�����{�}�T���ixս�
=��\)�aG����罏\)�u�ȴ9���P��\)���w�����C����-��xվ�㽺^5B1|B�B&wlBB0P�B�6A�3�BdJB�;Bo|B�B#OBRnB\B!��B�B��B$17B'RQB�aBB
�B�BB�B�1B[�B'HB*�(B�B�A��B�A��BZ�B=B˯B��B�.B"!B&��B۶BL#Bg�B!g�BYB'��B�tB�aB*އB-M�B�NBNB-�B#ճB
-�B`9BTYB �A��gB��B��B
�B
��Br�BB^�B�XB��B%@Bm�B��B��B��B�B��B55�B@B@VB&��B>�B0@�B=A��mBD�B��B�GB��B�>BAB��B!��BʄB�/B$K�B'�BD,B��B
�>B>�B?B�BA:B?�B*�B�NBAYB 0�BuA�{�B@+B?�B�B?�B�mB>�B&��B�TBAB;�B!��BA[B'��B?lB�B*�B. B��B
�TBC�B#��B
=�B@�B�JB �xA��)B��B@�B
7�B
H5B@8B@>BBhB��BغB?�B?�B�PB�B�B��B�qB50A�z1A=?�@�6qA��A[]A��Aص�@��@���Al�eAʿ6B	�N?�߻AV�A*��By"A��A��8A�+�A���A�b#A|�Az�A�7�AUh�>��C?���Ar�OA�EA��AA�>�A�{�B &AҠ�A�5�A�9A�t}@���@ƅ�A5+�A��NAz�CAO�)AE��A��@�ƼA��)A�&y@��A��B )�A��Ap�7@6��A��jB�NA �D@���A�)A�PA!�eA�=�A�RjA��aB�~BA�{�A���C��AZ�;=���?L�.A�a�C�>YC���AJߏA��LA= �@���A�(�A[�A�TmA؀N@�L@��AkNHA�kB	�?�:�AV�2A*�:B��A�w�A��`A��A�vfA�
�A{�AzJ�A���AU�>���?��An�"A���A��A�l=A�|�BDA�w�A��A�<A��H@�)P@���A6��A�Q�Az�AO4?ADg�A���@�ЫAׅ�A�}�@�DwA	�B E�A�y�Ap�k@3��AΑ:B��A �@юhA�|�A�3�ADA��A�0�Aņ�B�B7NAܵ�A�
C��+A[��>)��?H�A��JC�F C��AKX   $   0      
         
                     N   1   "         1            #   <   '                        )   ,   7      	   )   2   F               
   ,   F         !   F      
            )            ;         !         .            #   	      *   P      #                                       ?   '            1            /   )                     !   %      %   +               #                  5   /         #   '                              ?         %                                 )                                             !               -               !                     !   #         #                                 +   )         #                                 9         %                                 )   O��-Od��N�|�N�UN�VWN*�vN���NN�N3yO�w^N�8O��	Nd�VO�a�N�[�Ov��N$�N|GPH-UO9%NTwBN�UOW��O��TO�oN�{�N.�O�`�O��N`��O���O�V�OYڊN�F�O���Nn�O=nO��O�ɻO�HO>�9N�R(N�0nOHyxN�B�P��P"]N?\�NȒ�O�TfO>(MN	�O3mN/��Nf&N�MaO*��N���N�g{O;�P���Ng�qNe}�O�{N��FN���O"%$OK7+N?�O��OhbLN�FN"�ON�]P�QN$K`  D  0  �  �  P  ;  �  �  ;  -  �  �  t  �    o      e  4  �  �    ?  x  �  [  �    :    ~  �  k  N  �  �  �  	�  
}  �  �  5  �  �  �  �  �  5  g  
@  �  �  4    <  	�  �  �    h  �  W  �  �  �  	�    "  �  �     q  B  N  �;�`B���
�T���o�D�����
�ě��o�t��T���T���T���T���@��'�1��t���C����㼴9X��j�����#�
�,1��P�o��/��/�C����+�+�,1�aG��<j���t��t��49X��%��P��P�����#�
�H�9�<j�',1�'�\)�,1�,1�0 Ž49X�P�`�H�9�<j�@��@��L�ͽu�H�9�P�`�P�`�Y��u�e`B�aG��m�h�u�u��7L���w�����1����	!!
��������#<HU`c_UTH</#.0<IUYVUQI<20-......���

������������$**56CHKKICB6* "$$$$��������������������imz����zwrmlgeiiiiii56@BOUROGEEBB@635555?BIOTVTOHBA>????????������ !#���������������������������������������������NOU[ehkkha[XOONNNNNN#.5B[coy{tg[N5)#��������������������#)5BN[\glkf[NB5)' #!#&./00/#"!!!!!!!!�����
#$#
������$0<Ibn{�����{t<0$!$!)36A?65,)��������������������{���������������|{{��������������������������� ����������������
#$#"
�����NOY[chpmhhhhh[WROLNNdht���thcdddddddddd��������������������)5<=?;53)#��������������������^cmz���������zxxta^^��������������_aemz���������zmb]]_��������������������06BO[htz��yqh[YJ92.0��������������������������������������������������������������!),69;6)������:?FINU^bcdec`UI=<98:EHLUZaehhedaXTKHECCETUX`alnrnmnnnba]UTTTz������������}zzzzzz����������������������������������������<AIbn{������{bIG303<������ �����������459?BGKKGB5344444444��������������������������������KNW[fgtyzxtrg[NKEBDK��������������������)6BLS[[SQOHB?763351)����

�����������fgt��������vtgffffff������������������������������������������������������������_aalmtz��~zuma______������������������������(/8;A@7������nt�������vtlnnnnnnnnx�����������~txxxxxx5B[grogXTNB5������


�����������������������������������������������5<HUanz�}zonaUH@<35��������������������PUVanvz~zyqnaWUQOPPCHUakqrrppsniaUHB?@Cvz����������zytvvvv����������������������
#$# 
 ������
/;GISUP/#
������������������������������������������	���!����	�����ʾZ�M�A�4�3�4�4�8�A�M�Z�f�f�n�m�o�l�f�b�Z�������������ʼ˼ּ��ּϼʼ�����������ŔœōŔŚŠŭŲŸŴŭŠŔŔŔŔŔŔŔŔ�	������������	��!�"�,�+�"��	�	�	�	�#����#�/�;�2�/�*�#�#�#�#�#�#�#�#�#�#�6�/�-�6�:�B�K�J�O�[�_�[�O�B�6�6�6�6�6�6�S�Q�F�E�@�F�S�X�_�l�x�z�x�l�c�_�S�S�S�S�����������������������������������������m�`�W�Q�M�N�T�`�r�y�����������������z�mÇÄ�z�o�zÇÓàìôõìàÓÇÇÇÇÇÇ���������������$�0�9�7�9�5�:�:�2�$��e�c�Y�M�Y�]�e�r�x�~���~�r�r�e�e�e�e�e�e�׾ʾ��������ʾ���	�� �����	����׽ݽڽнĽĽ��ĽŽнݽݽ�����������ݽ��=�7�2�1�0�0�0�<�I�V�b�o�q�o�l�d�b�V�I�=�<�9�<�H�O�U�a�c�b�a�U�H�<�<�<�<�<�<�<�<�����������������������������������������g�Q�F�A�9�7�:�B�g���������������������g²§¦¦²¿¿����������¿²�����������������������������������������Ŀ������Ŀ˿ѿܿݿ�����������ݿѿĿĿ������������Ŀѿݿ���������ݿѿ����5�(��(�5�A�N�g�s�������������s�g�N�A�5��ݾ׾ξʾȾžʾ׾���������������ù¹��������ùϹܹܹ߹����ܹعϹùú����$�'�(�1�.�'�������������|�k�d�a�m�y���������ĿǿпտӿĿ�����������������������������������������������
����)�5�B�N�B�;�5�)��������s�g�V�K�Q�g�s�������������������������s�N�A�(� ��(�A�N�Z�g�q�x�������������s�N��ƺƳƫƥƤƧƫƳ����������������������������������������������������������w�s�t�r�r�p�s������������������������ìéãçìù��������ÿùìììììììì�a�\�Z�^�a�m�u�z���������������}�z�m�a�a�����������������ûܻ���� �������ܻл������������������'�.�.�5�4�.�'�������������������(�4�B�F�A�@�4�(����������5�A�N�Z�`�g�g�Z�S�A�(���ѿϿĿ��������Ŀпѿտݿ޿����ݿѿѾ����������������ƾʾѾӾϾʾ������������f�Z�V�X�Z�b�f�s��������������������s�f������������������������������������������л����������ûм��4�F�M�8�4�������������)�B�[�tĚĮĪĚč�t�[�B�)��������������������	������������������Y�T�Y�b�f�r�������������������r�f�Y�Y�ʼ��Ƽռڼ������!�3�7�5�!�����ּ���������*�6�C�L�O�W�Y�S�O�C�6�*���
�������&������������m�d�n�y�����������������������������y�m�ɺƺ������ºɺֺ�ݺֺ̺ɺɺɺɺɺɺɺ�����ÿûü�������������������������������I�A�=�6�5�=�I�V�b�h�o�p�o�h�b�V�I�I�I�I�ּռʼżü����ȼʼּܼ��������ּ'�"��&�'�4�@�M�V�Y�\�Y�R�M�@�4�'�'�'�'����������������������������������������ĿļĸĺĿ����������������������������Ŀ�y�G�3�1�:�S�`�����Ľн�����ѽ����y���������������������������������������ŠŗŠţŠŝŠũŭŹ��źŹŭŠŠŠŠŠŠ�U�H�?�:�<�9�/�0�<�H�M�a�nÏÒÑ��n�a�U����������������������������������������������������������������������h�c�[�Z�Z�[�h�tāčĚĦĨīğĚčā�t�h���z�r�k�l�o�v�z������������������������E�FFF$F1F5F3F1F$FFFE�E�E�E�E�E�E�E��������������	���"�$�(�"� ��	���������������������ùϹܹ����ٹù���������������������������������������!������������D�D�D�D�D�D�D�D�D�D�EEEEED�D�D�D�D�EPE*EE E*E;ECEPE\EuE�E�E�E�E�E�E�E�EuEP����������������������������������������   ? 4 , A Y F f A G u V - - 6 1 H w * + 2 1 i U  A 6   & � N 3 ' , = ( L $ O B i M V 6 < 2 I a ] X  g Z = n ' d % * ! E ) G J 8 \ A F p  @ S 1 B ; D  D  �  �  �  �  e  �  �  J  �  �  �  f  	  �  �  U  �  Q    U    
  x  H  �  K  z  A  �  m  >  �  �  �    /  �  S  a    �  �  �  '  �  �  W    �  �    B  N  �  �  �  �  �  �  ^  q  x    �  �  o  �  �    �  �  2  �  �  B  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  �  	  &  :  D  C  =  3  #    �  �  �  �  �  X  
  �  
  �  �  �  
  $  /  '      �  �  �  ]    �  �  y  =  �  U  �  1  F  Z  m  |  �    �  �  �  �  �  �  �  �  n  J    �  ;  �  �  �  �  �  �  q  Z  B    �  �  |  ?  �  �  b        =  H  N  H  ?  6  ,  !        �  �  �  �  R     �   �   N  ;  9  7  5  5  4  6  :  >  I  V  c  q    �  �        �  �  �  �  �  w  z  �  u  ^  E  +    �  �  �  �  �    T  *  �  �  �  �  �  �  �  �  �  �  �  �  �  }  j  W  C  .      '  -  4  :  =  @  C  =  2  (        �  �  �  �  0  �  a  )  ,  )      �  �  �  �  v  ]  T  D  "  �  �  W    �  ~  �  �  �  �  �  v  T  1    �  �  �  f  5    �  �  Y    �  �  �  �  �  �  �  �  �  �  �  n  C    �  w  9  B  |  W  "  t  d  Q  ;       �  �  �  q  A    �  �  a  "  �  �  k  7  �  �  �  6  Z  z  �  �  �  �  �  �  �  �  L    �  �  �   �    Z  �  �  @  �  �  �  �  �      �  �  F  �  o  �  �  C    E  a  l  k  ^  C    �  �  �  ?  �  �  "  �  B  �    �  �  �     
        �  �  u  c  Q  ;  "    �  �  �  p  B        �  �  �  �  �  �  �  �  }  k  Z  H  <  3  )      ]  b  S  ;    �  �  �  j  <    �  �  m  1  �  �  :  �   E  4  .  (  !        �  �  �  �  �  k  6  �  �  J  8  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  q  i  a  X  P  H  �  �  �  �  �  �  �  �  �  �  �  �  `  7    �  �    @  �  -  H  |  �  �  �  �  �  �    �  �  �  x  2  �  �  ^  �    �  �    '  2  <  #  �  �  �  �  Y    �  �    _  h    �      =  S  k  x  q  d  P  ;    �  �  �  |  G  �  �  +  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  L  '  �  �  ^  �  [  U  O  I  A  9  /  %        �  �  �  �  �  h  =    �  �  �  �  �  �  i  Q  :  %      �  �  �  �  �  �  a  &   �  �  �  �       �  �  �  �  �  �  w  a  B    �  Z  �  G  b  :  3  ,  %          �  �  �  �  �  �  �  �  �  u  e  U        
    �  �  �  �  �  �  �  �  �  f  B    �  �  E  {  }  y  o  `  J  -    �  �  c  *  �  �  �  q  1  �  y   �  �  �  <  Z  �  �  }  n  V  6  	  �  �  ?  �  J  �    �  b  �  �    (  B  T  `  ^  R  f  i  N  &  �  �  b  �  !    �  �  �    5  J  K  8       �  �  �  h  E    �  �  �  �  -  �  �  �  �  �  �  �  �  �  �  �  ]  1     �  �  A  �  z    �  {  g  V  G  9  +        �  �  �  �  �  �  �  �  �  �  �  �  o  Y  C  *    �  �  �  V  .    �  �  1  �  �  �  �  	S  	u  	�  	�  	q  	B  	  �  �  Y    �  3  �    j  �  %  �    	�  
  
.  
C  
T  
b  
m  
z  
{  
l  
7  	�  	R  �  2  �  �  8  {     �  �  �  �  �  ~  d  E    �  �  �  J  	  �  �  ;  �  �  �  �  �  �  �  o  U  9      �  �  �  �  ^  ;    �  �  �  �  5  '      �  �  �  �  �  �  �  �  �    c  G  )  
   �   �  �  �  {  m  ^  O  B  8  -      �  �  �  �  d  ,  �  �  f  �  �  �  �  �  �  �  �  �  �  �  w  r  n  m  k  b  Z  Q  I  z  �  �  �  �  �  �  �  �  {  n  s  c  T  =     �  �  {  �  �  �  �  �  �  ;  �  �  y  )  �  �  8  �  M  7  �  M  �  }  �  r  `  M  8  $    �  �  �  l  /  �  �  �  m  =     �   �    #  4  *      �  �  �  �  �  �  k  Q  ;  '      �  �  g  S  B  :  '    �  �  �  �  m  G    �  �  W  �    O  �  �  y  	   	Z  	�  	�  
  
6  
?  
1  
  	�  	�  	9  �  �    �  !   �  �  �  �  �  {  q  f  [  O  D  9  -  !    
  �  �  �  �  �  �  �  y  \  >    �  �  �    _  L  L  B  '     �   �   �   �  4  +  !      �  �  �  �  �  �  �  v  d  M  0    �  �  �                    �  �  �  �  �  }  ]  =     �   �        �  :  3  )    �  �  �  �  Q  �  p  �    -     �  	H  	�  	�  	�  	\  	.  	  �  �  Q    �  8  �  E  �  +  �    �  �  �  }  e  K  /    �  �  �  s  C    �  �  �  b  A  1  ;  �  �  �  �  �  �  �  s  `  H  0      �  �  �  �    (  ?    �  �  �  �  �  �  �  �  �  �  �  �  �  n  P  :  &  
  �    h  W  9    �  �  2  �  ^  �  v  Z  '    �  �  D  �  '  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  e  �  �  )  �  W  Y  [  ]  Y  T  O  H  A  9  4  0  -  +  +  +  +  *  )  (  �  �  `  2  �  �  l  $  "     �  q    �    �  �  �  �  �  �  �  �  ~  n  [  E  *  	  �  �  �  L    �  g    �  @  �  �  �  �  k  Q  7    �  �  �  �  z  V  2    �  �  �  �  �  	L  	f  	y  	�  	  	d  	B  	  �  �  �  O  �  �    7  ]  r  g    �  �  �    �  �  �  �  �  �  �  �  a  <    �  �  ]    �  "    �  �  �  v  C    �  �  [    �  �  :  �  �  \     �  �  �  w  ]  4    �  �  �  �  �  b  9    �  �  �  �  �  �  �  l  M  )  �  �  �  D  �  �  [  	  �  4  �     z  �  
  Y           �  �  �  �  �  �  {  ^  >    �  �  �  Y  1  	  q  �  �  �  �  �  �  �  �  �  �  {  m  _  Q  B  0    �  f  �    B  4    �  �  �  M  �  |    
z  	�  	  O  �  �  ~  X  N    �  �  �  �  [  %  
�  
�  
<  	�  	W  �    8  _  e  %  �  �  �  �  y  U  2    �  �  �  p  J  #  �  �  �  |  B     �
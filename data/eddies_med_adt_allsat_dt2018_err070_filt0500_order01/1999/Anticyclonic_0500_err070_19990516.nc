CDF       
      obs    5   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��G�z�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�J   max       P�p�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       %@     max       =��      �  T   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��
=p�   max       @E�G�z�     H   (   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��\)    max       @v~�Q�     H  (p   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @N            l  0�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @���          �  1$   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       <e`B   max       >�ƨ      �  1�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B+��      �  2�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B+J�      �  3�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�:   max       C���      �  4t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?���   max       C��-      �  5H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  6   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          C      �  6�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -      �  7�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�J   max       P/�u      �  8�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�GE8�4�   max       ?��J�M      �  9l   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ;o   max       >%�T      �  :@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @E�33333     H  ;   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��\)    max       @v~z�G�     H  C\   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @!         max       @N            l  K�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�@          �  L   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         Fp   max         Fp      �  L�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�kP��{�   max       ?��J�M     �  M�                  |         )         '      
         b   !   !         +                        
   
               #         
   $            !         �   
   G   	      &   .O	�N4Oj��N���O+�DP<b6O��rNcJOO�BqNX�N].�O|��N�c�N� M�JNE�
P�p�OXt�Oگ�N-�N2��O��OA�hNK�N��1N^.JO=AN���N#�%N��kOE�N?�Nݸ�O*ORN�o�O&~�OR@P/�uOF�P#u�O�O*�O+�O�<}N.�Nʀ�O�T�NRz�O�j�N�[N�(HO%ԜO���%@  ;o;D��;ě�<t�<t�<t�<t�<t�<t�<49X<49X<T��<T��<u<�o<���<��
<�9X<�j<ě�<���<��=t�=�P=�w=#�
=#�
=#�
='�=,1=,1=0 �=8Q�=<j=@�=H�9=H�9=L��=P�`=P�`=P�`=P�`=Y�=aG�=y�#=�o=��P=��P=�{=Ƨ�=���=��������������������������������"#,06<IUbddbZTI<0,$")6BBBB?6))���	"/373/$"	��06BN[g{�������tg[N:0bcjmt}�����������tgbcciinrzz�~zncccccccc}vu���������������}��������������������)6BMGB6,)�������������������������������������������� 
!#+,'%#����������������������)-)'	��� 5K[ef\V[K)���#)5BN[]goqng[NB75/)#	*/<GPUWUVNH</	��������������������61))66666666�~�������������������������������������V[^hmtxxtph[VVVVVVVV�����


���������#%/<AHJH</#+/;HST]`^YTHE?;92/++ ##-/6<HGF><0/#   #*/<=<63/#        ������������������
#,)&#
�����

�������#/6<GHLHC</'#@<?BDIO[hlqutrmh[OB@)366864,)CFCETUanz}~|zxnaUPHCmmotz������������zqm����������������������������		
��������*5=;1)����)*/3575/)4125BHNU[^ghgc[ZNB54��������	

�������������������������)+6BCB6)))')))))B98<BO[ad`[VOBBBBBBB��������

����! )69@BHB6+)!!!!!!!!HR[hn�����������thXH����������������������������������������

�����zutv~��������������z��������%����������������������ù������������������ùñîøùùùùùù��������������������������r�e�b�b�f���àãàÞÜÞÓÓÑÇ�z�o�z�z�~ÇÓÚàà�a�m�p�r�w�z�����z�m�a�[�H�C�;�C�H�T�\�a�)�B�[�i�p�k�[�H�6�����������������)�a�m�z�����������������z�m�a�T�H�H�K�T�a������������������������������������޾(�A�M�f�k�m�k�a�Z�O�M�A�/��������(�����������������������������������������ּܼ��ּʼ����ʼϼּּּּּּּּּ�ù��������������ùìàÓÇÄÄÖÝàíù�
��#�/�7�<�8�3�/�#���
����
�
�
�
�l�n�x���������������x�l�_�Y�_�f�l�l�k�l�����ûлֻлû��������������������������M�Z�[�`�b�`�Z�O�M�H�M�N�M�M�M�M�M�M�M�M�������$�I�@�0�����ƳƎ�g�\�8�9�\ƁƧ�����������������������������������������޾�������׾�߾ʾ���������s�o�l�k�f�s��`�m�x�y�������y�m�b�`�`�`�`�`�`�`�`�`�`�Z�N�H�A�A�=�A�N�O�X�Z�_�Z�Z�Z�Z�Z�Z�Z�Z�����������������������ýóòù�Ҿ��(�2�5�A�C�C�M�A�7�(���� � ��
��@�B�L�U�Y�\�Y�L�C�@�:�?�@�@�@�@�@�@�@�@�B�N�V�N�M�H�B�5�)�(����)�5�6�B�B�B�B  �#�&�,�*�#�"��
��������������
�� �#�#�����������������������������������������N�[�a�g�j�q�g�c�[�N�J�K�N�N�N�N�N�N�N�N�s�������������������������s�m�s�s�s�s�Z�f�o�s�����v�s�r�f�Z�M�I�A�>�<�A�N�Z���������������������������������������������������������������������-�:�F�M�S�_�h�k�g�_�S�F�B�:�-�(�%�%�)�-�r�~�����������������~�w�r�e�a�b�e�g�r�rFFF$F1F=F>FBFHFDF=F1F$FFFFFFFF�����������������������������������������	�/�;�T�m�{��u�m�a�T�-������������	����	��"�.�;�G�H�D�;�.�"��	���ھ�����(�<�M�e�g�N�5������տҿݿ��������������ĿϿǿȿĿ����������������������`�m�����������������z�y�x�m�a�`�T�V�_�`�/�<�H�U�a�f�k�k�a�]�U�H�<�8�/�)�#�#�,�/���������ǽؽٽн��������y�q�n�m�r�z�����@�3�(�(�'�&����� �'�3�6�@�A�@�@�@�@�Ŀѿݿ�������ݿѿɿĿ��ĿĿĿĿĿĿ�D{D�D�D�D�D�D�D�D�D�D�D�D�D�D�DvDqDrDqD{�M�Y�a�\�Y�M�F�@�<�;�@�A�M�M�M�M�M�M�M�M�4�@�F�R�Y�j�n�f�R�M�@�4�'�����#�'�4�T�\�a�c�a�X�T�H�F�D�H�L�T�T�T�T�T�T�T�T�
���#�0�7�I�J�J�I�<�0�'�#���
�� �
EuE�E�E�E�E�E�E�E�E�E�E�E�E�E}EuEtEmEjEu�������ʼּڼ������ּʼ�����������  I 2 e > ) B L  ^ d W 8 7 l l F @ ( ` x 6 ) n L ` J : [ x # T \  ( ; C Z c  % 4  : s  ) U D O u 3 7    +  �  �    z  0    �  �  d  �  =    �  4  �  �  �  �  c  �  W  �  i  �  �  =  '  \  �    4    e    x  �  R  �  �  F  +  g  �  x  �  �  y  [  M  8  q  !<�o<e`B<���<���=o>1'<�`B<e`B=L��<u<���=L��<���<�j<�C�<��
==T��=]/<���<�`B=�O�=�P=0 �=#�
=L��=P�`=Y�=0 �=P�`=P�`=8Q�=m�h=�+=m�h=��T=�\)=��
=u=�-=y�#=�%=�+=� �=}�=��>�ƨ=�1>t�=��=�;d>!��>-VB�5B��B&�uB?A���B	e�B
Y=BψB 48B /BB�B!��B��B$��B#A?B��BB�BZ�B9�B ��B�BYBȋB��BCvBq�A�VvBD�B.�B'�B$�2B$B?B(�B+�B�MBI-B K�B�zB!dB��B>hBOB�IB+��Bm�B��B�B�B�HB��B	�B��B�"B� B �B&BwB��A���B	�B
@�B��B ?�B =\B�^B!�"B��B%>�B#4KBNGB��BE=BKB �ZB,0B?�B��B�=B@�B�A���BB�B-)B?�B$R�B$=�B�	BA�B֔BD4B <�B�B7B�dBޗB>�B@�B+J�B=�B��B?TBL�B�BH�BBcBҦB�A�}zA�Q�@� �A�M�A�]�A�	A�g5A��"A9d;@�]A �zA��hA�d�@��_@� hA>L�BxA��AJ;1Al4A��0A�+�A6:�?���A���A�lCA���A��*A���AG�A?_AKF�A�>�@��~@j6C���A��A�ߩA\��A���Au�Ak�!AĄhA �b?�:A|��C��@���@�jQA�C�A��=C�  @���A��A�~�@� �AɅkA�UAՃ�A�{�A�E~A:Ej@�h�A �A�}A�@��@� VA>�PB
?A�o�AI4Am�PA� �A�q!A5l?���A�*0A��<A�*A�j�A�xOAF�A>�GAJ��A҃0@���@|C��-A��*A���AZ=�A���At�CAj��AĄ�A"��?���A|��C�ٖ@�Q�@�ȚA��A� C��A Z                  }         *      	   (               c   !   !         ,                           
               #             %            "         �      G   
      &   .                  -         !                        C      #                                                         -      )            !                                             #                                 !      !                                                         -      )                                       N���N4Oj��N���N�� O�D�O�ZNcJOO,��NX�N].�O_`�N�c�N��M�JNE�
O�pEO)*3O��~N-�N2��Ob*QOA�hNK�N��1N(8�O=AN���N#�%NIN�qN?�Nݸ�O�PN��N���OR@P/�uOF�P#u�O�O*�O+�O�	>N.�Nʀ�O�JNRz�N���N�[N�(HO�O���    �  _  �  8  �  �  �  �  $  :  A  �    �  F  Z      �  g    �  a  e  �  �  �  �  [  �  f  p  K  T  �  �  h  �  ;  <  e  h  4  �  �  �  �  
�  1  %  	�  
�;�o;o;D��;ě�<�C�=T��<e`B<t�<�/<t�<49X<e`B<T��<e`B<u<�o=��<���<���<�j<ě�=\)<��=t�=�P='�=#�
=#�
=#�
=,1=49X=,1=0 �=<j=@�=Y�=H�9=H�9=L��=P�`=P�`=P�`=P�`=}�=aG�=y�#>%�T=��P=�
==�{=Ƨ�=��#=��������������������������������"#,06<IUbddbZTI<0,$")6BBBB?6))		"///,%"					ADN[gt��������tg[QIAnhiptu�����������tnncciinrzz�~zncccccccc����������������������������������������)6BMGB6,)����������������������������������������
#*+&##"
��������������������)-)'	�����)5==<5)�3/5BCN[aglmgf[NCB:53#./<DNSUUSSH<#��������������������61))66666666����������������������������������������V[^hmtxxtph[VVVVVVVV�����


���������!#)/<@HIH</#!!!!!!!!+/;HST]`^YTHE?;92/++ ##-/6<HGF><0/#   #*/<=<63/#        ��������� �����������
#''$#
	��������

�������#/6<GHLHC</'#>@BEJO[hlqttqlh[OCB>)256763*)!GHJUZanzz|zzunaUJHGGmmotz������������zqm����������������������������		
��������*5=;1)����)*/3575/)4125BHNU[^ghgc[ZNB54��������	

�������������������������)+6BCB6)))')))))B98<BO[ad`[VOBBBBBBB��������

�����! )69@BHB6+)!!!!!!!!jlrt������������{tjj����������������������������������������

�����zutv~��������������z������� ������������������������ù������������������ùñîøùùùùùù��������������������������r�e�b�b�f���àãàÞÜÞÓÓÑÇ�z�o�z�z�~ÇÓÚàà�T�a�f�m�q�u�n�m�k�a�T�O�H�F�H�T�T�T�T�T�)�B�U�^�[�R�B�6�)�����������������)�T�a�m�z�������������z�n�m�a�T�P�N�P�T�T������������������������������������޾4�A�I�M�W�Z�\�Z�X�M�A�4�(������(�4�����������������������������������������ּܼ��ּʼ����ʼϼּּּּּּּּּ�ù��������������ùëàÓÇÆÐØàìðù�
��#�/�7�<�8�3�/�#���
����
�
�
�
�x���������������x�l�_�^�_�l�l�n�x�x�x�x�����ûлֻлû��������������������������M�Z�[�`�b�`�Z�O�M�H�M�N�M�M�M�M�M�M�M�MƳ��������������������ƳƚƍƉƎƑƞƳ���������������������������������������Ѿ������ʾ׾پھʾ�����������r�o�m�s����`�m�x�y�������y�m�b�`�`�`�`�`�`�`�`�`�`�Z�N�H�A�A�=�A�N�O�X�Z�_�Z�Z�Z�Z�Z�Z�Z�Z����������
���������������ùþ�����޾��(�2�5�A�C�C�M�A�7�(���� � ��
��@�B�L�U�Y�\�Y�L�C�@�:�?�@�@�@�@�@�@�@�@�B�N�V�N�M�H�B�5�)�(����)�5�6�B�B�B�B�#�&�,�*�#�"��
��������������
�� �#�#�����������������������������������������N�[�a�g�j�q�g�c�[�N�J�K�N�N�N�N�N�N�N�N�s���������������������s�o�s�s�s�s�s�s�Z�b�f�r�o�l�f�Z�M�L�A�@�?�A�M�V�Z�Z�Z�Z���������������������������������������������������������������������:�F�K�S�_�f�k�f�_�S�F�F�:�-�)�&�&�-�.�:�r�~�����������������~�y�r�e�c�e�f�h�r�rF$F*F1F:F=F@FDF>F=F1F'F$FFFFFFF$F$�����������������������������������������	�/�;�T�m�{��u�m�a�T�-������������	����	��"�.�;�G�H�D�;�.�"��	���ھ�����(�<�M�e�g�N�5������տҿݿ��������������ĿϿǿȿĿ����������������������`�m�����������������z�y�x�m�a�`�T�V�_�`�/�<�H�U�a�f�k�k�a�]�U�H�<�8�/�)�#�#�,�/���������Ľ̽ͽƽ��������y�u�s�y���������@�3�(�(�'�&����� �'�3�6�@�A�@�@�@�@�Ŀѿݿ�������ݿѿɿĿ��ĿĿĿĿĿĿ�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��M�Y�a�\�Y�M�F�@�<�;�@�A�M�M�M�M�M�M�M�M�4�@�M�P�Y�Y�Z�Y�R�M�@�4�/�'�&�%�'�+�4�4�T�\�a�c�a�X�T�H�F�D�H�L�T�T�T�T�T�T�T�T�
���#�0�7�I�J�J�I�<�0�'�#���
�� �
EuE�E�E�E�E�E�E�E�E�E�E�E�E�E}EuEuEnElEu�������ʼּڼ������ּʼ�����������  I 2 e 8 , ( L  ^ d V 8 @ l l 1 8  ` x 2 ) n L a J : [ v  T \  ) 1 C Z c  % 4  : s   U / O u 2 7    �  �  �    �  �  >  �  h  d  �  �    �  4  �  �  {  �  c  �  �  �  i  �  �  =  '  \  �  �  4    Q  �  $  �  R  �  �  F  +  g  A  x  �  F  y    M  8  e  !  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  Fp  �  �  �            �  �  �  �  �  U  !  �  �  �  p  D  �  �  �  �  �  �  �  �  p  V  <  !     �  �  �  v  P  )    _  ]  X  S  L  B  3      �  �  �  z  O      �  �  }  6  �  �  �  �  �  p  ^  P  >  )    �  �  �  ^  %  �  �  k  *  �  �  �    %  0  6  8  4  *    �  �  �  e    �  >  �  N  �  �  �  >  z  �  �  �  �  �  T  �  g  �  
�  	�  �  0  [  �  y  v  w  z  |  �  �  �  {  i  V  B  0  !        �  �  �  �  �  �  z  h  V  C  0  $           �  �  �  �  }  R  '  �  �  !  O  w  �  �  �  �  �  �  �  �  o  1  �  a  �    I  $  0  <  H  G  D  A  B  F  I  P  [  e  n  q  u  v  l  b  X  :  7  5  5  6  5  *         	    �  �  �  �  �  �  d    ,  >  <  $    �  �  �  �  �  w  R  '  �  �  9  �  c  z  a  �  �  �  �  �  �  �  �    z  u  q  X  9    �  �  v  %  �  �  �            �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  k  ^  Q  D  7  *    �  �  �  �  �  u  Z  ?  $  F  C  @  =  :  6  0  *  #          �  �  �  �  �  �  �  Z  �      (  (  1  1  1  <  Q  V  *  �  �  �  @  �  y  �  �  �        �  �  �  �  t  @    �  x  '  �  x  �  "                �  �  �  �  j  :    �  �  6  �       �  �  �  �  �  �  �  �  �  �  �  �  p  `  P  @  /       �   �  g  `  Y  R  L  F  F  F  F  F  C  >  9  3  .  0  4  9  =  A  �  �  �      �  �  �  �  N    �  �  E  �  �    �  �    �  �  �  �  �  �  �  �  �  }  q  n  l  h  e  a  Z  E  0    a  j  r  u  j  _  S  H  =  1  %        �  �  ~  +  �  m  e  ]  U  M  E  =  5  +  "         �   �   �   �   �   �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  �  �  �  �  �  �  l  X  S  G  -    �  �  �  �  E  �  �  Q  �  �  �  �  �  s  d  T  C  1    �  �  �  R    �  p  0   �  �  �  �  �  �  {  v  k  ]  N  @  2  $    �  �  �  �  �  �  D  P  [  ]  [  I  4    �  �  �  |  K    �  �  c  &   �   �  �  y  m  w  �  ~  t  d  R  >  *    �  �  �  �  v  O  !   �  f  f  f  f  f  f  e  `  Y  Q  J  C  ;  -     �   �   �   �   �  p  p  j  ]  G  (     �  �  [  !  �  �  j    �  D  �  `   �  G  K  F  A  ;  /    
  �  �  �  t  C  
  �  b    �  X  �  N  R  Q  K  >  0    	  �  �  �  �  }  [  5  
  �  �  _    p  Q  5  O  �  �  ~  j  R  ;    �  r  �  X  �  /  U  a  X  �  �  �    c  E  '  	  �  �  �  r  8  �  �  y  :  �  �    h  \  ?    �  �  �  r  ;     �  �  @  �  �  L  �  �  D    �  �  �  �  �  �  �  �  �  |  d  G  $  �  �  �  ^     �   �  ;  1    �  �  �  �  �  p  _  N  <    �  �  a    �     �  <  ;  :  5  .       �  �  �  �  �  �  x  U  .  �  �  _   �  e  [  W  ]  \  X  J  9  '    �  �  �  r  ,  �  �  �  u  K  h  K  ,    �  �  �  p  C    �  �  �  J    �  W  �  Z  1  �       ,  1  4  0      �  �  ^  	  �  9  �  >  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    E     �  �  �  �  �  �  s  _  K  6      �  �  �  ~  R  !  �  �  �  �  �  �  �  H  �  >  �  �  �  �  a  �  �  �    �  S  }  
'  �  �  �  �  �  �  �  l  U  9    �  �  �  �  y  K    �  �  	�  	�  	�  
  
:  
]  
}  
�  
�  
�  
�  
[  
  	�  	U  �  1  O    �  1  &      �  �  �  �  �  �  �  �  s  W  Z  `  q  �  �  �  %  $    �  �  �  �  o  D    �  �  g  !    �  y  $  �  w  	�  	�  	�  	�  	�  	�  	r  	,  �  �  9  �    !  �  +  y  �  �    
�  
<  	�  	�  	�  	X  	$  �  �  u  1  �  �  q  (  �  +  �  �  '
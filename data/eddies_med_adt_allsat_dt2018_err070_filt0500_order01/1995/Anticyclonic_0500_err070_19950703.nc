CDF       
      obs    <   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?Ƈ+I�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�-k   max       P�+G      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��   max       >%      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��R   max       @F���Q�     	`   |   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @vw�z�H     	`  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @P�           x  3<   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @���          �  3�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��j   max       >;dZ      �  4�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��N   max       B3��      �  5�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�~A   max       B3ª      �  6�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?	�X   max       C�|@      �  7t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?��   max       C�o�      �  8d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  9T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          I      �  :D   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          )      �  ;4   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�-k   max       P �      �  <$   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�u�!�R�   max       ?�7��3�      �  =   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��   max       >%      �  >   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?5\(�   max       @F���Q�     	`  >�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�     max       @vw�z�H     	`  HT   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @P�           x  Q�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��@          �  R,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @�   max         @�      �  S   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?!-w1��   max       ?�64�K     �  T      
                  �      H   	               7      #            $   E                      (      $               2                        )         �         )      H            &      B   N\N�1GO\��O�$OBƼOW��N VPn�NNTP�+GN�k�OBQ�N��O�'O�P}ѦO@�O���NƩ�N9��O:��O�O(PlμN���O k�O�C�O��O_�UOc"�O�4N{�MO�]6O��M���O!�	NS��PJ[~N�D�N�%�N��Ok�OmzN�nORi�O��On(�NךOO��
Nw�O��O�b9N��jO��;N³@O P�N4��O��N�8KO��M�-k����j�u�e`B�t��o���
��o��o:�o:�o:�o;�`B<t�<#�
<49X<49X<49X<D��<e`B<e`B<u<�o<�o<���<��
<��
<��
<�j<���<�/<�`B<�h<�h<��<��=C�=C�=t�=t�=��=�w=49X=H�9=Y�=Y�=Y�=]/=e`B=m�h=q��=u=�7L=���=�{=�E�=�E�=�j=Ƨ�>%����������������������������������������B?HU[anz������znaUHBKHN[gt���������tg[RK��������������wornqqsz����������zw��������������������/4:B[g��������}g[G7/#-/172/#����)<F[|{g\5)������������������������>@KO[tx�����yth[ZOB>��������������������)5BIIRPB<;71)����)+-*'# ��l���������������zynl������������������������������������������������������������),,*)#/<>HMTUF</)#"!�����������������������������������������������������������������������������/31-68Qgt����uj[B5/������!(+0480)���64347BOZ_abec`[OGB96��������

����������
/<@IH</
������������������������������������
#/0442/(#
��~���������~~~~~~~~~~ #%0<IJKIEEC<10#!#*054<20#!!!!!!!!�")5BN[^[NH5��VX[^chtx���ytjh[VVVV,/<HUYaha]UH><0/,,,,-)/<>?><8/----------���������
���������������������������^bny{����{nb^^^^^^^^��������
�������,)'#,07;HT^beeaTH;/,nikt�������������ztn������������������������������

����FHUXakkeaWUOKHFFFFFF�����������������������������
��������������������������������� ���������ABCLO[hnoih_[OIBAAAA������

��������)*+,)!&)6BOY\WUNB>3,)�����
����)
)8@DED:6,)���������������������$�0�=�>�=�4�0�$�����$�$�$�$�$�$�$�$�G�S�`�l�y�{�y�l�f�`�S�G�<�E�G�G�G�G�G�G������������������������������������������������������������������������������������)�/�)�'�����������������������E�E�E�E�E�FFFFE�E�E�E�E�E�E�E�E�E�E��׾ؾ����׾׾Ӿʾʾʾоվ׾׾׾׾׾��6�hāĕėēć�t�[�M�B�6�'���������6D�EEEEEED�D�D�D�D�D�D�D�D�D�D�D�D��<�I�b�{ųŲŠŇŇ�n�b�D�ĽĸĿĿ���
�<�*�6�C�O�T�V�W�O�C�6�-�*�(�(�*�*�*�*�*�*���������������������������{��������������"�.�8�1�.�%�"���	����������	�
������"�;�G�X�`�e�`�T�;�.��	����׾оӾ��������������������������������������������	�"�-�7�/�	�������v�������r�z�������(�5�A�N�X�Z�g�n�g�^�Z�J�A�5�(�����(�(�5�N�R�Z�c�d�c�Z�W�N�A�5�(������(¿������������������½²¦¤£¦¯²½¿�Z�Z�b�]�Z�M�A�=�A�A�M�X�Z�Z�Z�Z�Z�Z�Z�Z��������������������������������������ûɻһɻû������������}�o�p�{�������B�[āđā�h�[�S�)�������ãà���뼱�����ʼʼʼʼ��������������������������#�/�<�G�H�A�<�5�5�/�#��#�%�#����!�#�ѿݿ�����(�0�;�;�5�(������߿ɿſѾ��	�"�.�G�T�^�`�T�G�;�.�"�	����������ʾ׾������׾ʾ�������}���������ʾA�M�Z�f�s��������s�\�M�A�5�*�+�1�4�7�A�.�;�T�m�������������o�`�;�.����� �.���(�1�(�%����	��	�����������*�6�C�O�\�k�g�O�C�*�������������������������ݿ׿ܿݿ���������������������������������������������������������������������q�f�d�r��ּ������������ּͼѼּּּּּּּ���������0�:�A�@�$�����ƣƚƍƂƂƐƧ���-�:�B�F�S�X�U�S�G�F�F�:�6�-�+�$�-�-�-�-�(�.�2�,�+�(������
�����(�(�(�(�f�s�~�w�s�f�Z�P�Z�e�f�f�f�f�f�f�f�f�f�f�a�l�t�y�������������y�l�`�S�Q�Q�R�S�]�a��(�4�A�E�I�B�A�4�,�(��������������������������������������������ʼռּ޼����ּʼ�����������������Ŀ���������
�����
����������ľĻĻĿ��"�/�5�8�<�;�1�"��	�������������������������������������������������������D{D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DD{DwD{���������������s�q�s�u���������������������(�4�A�M�S�Y�W�M�A�4�*�(���������������
��1�.� �
��������¿´¾·¿���z���}�z�u�n�a�U�J�U�V�a�j�n�y�z�z�z�z�ܹ�������!�!����Ϲ��������ùϹܼf�n�r���������r�f�c�Y�P�O�Y�b�f�f�f�fE�E�E�E�E�E�E�E�E�E�EuEqElEjEuE{E�E�E�E���������������������������������������������� �-�5�:�4�'������ܻлû��Ż���������������ֺɺƺɺֺֺ��@�L�V�e�r�������������������~�r�Y�T�L�@E7ECEPEREQEPECEBE7E6E7E7E7E7E7E7E7E7E7E7 K d W . R F b , A ; ( I D ^ O D r " U ; 5 < T d z b * ` 2 P T < 5 _ 7 8 M T Z u . . � H 0 < ; # W Q F z 5 4  M ` \ . O  d  �  �  2  �  �  i  �  :  �  �  �  �  *  
  h  g    �  b  �  >  A  �  n    �  C  �  \  �  �  2  E  c  i  �  �      �  B  �  �  G  �    K  �  \  �  �  �  �    c  q  �  �  ��j�e`B�D��<D��;o<t��D��>0 �;o=�t�<#�
<D��<D��=o<�=��<�C�=<j<�h<�t�=0 �=L��=�1<�j=#�
=#�
=T��=#�
=@�=��=o=�%=�w<��=@�=t�=���=#�
=0 �=��=ix�=aG�=L��=��
=��=�\)=y�#>;dZ=y�#=���=���=�Q�>O�=�-=�/=ě�>%=�S�>%�T>+B��Bh3B�B	�$B5NB¾B�B	5�B�8B��B?�B��B�B1B��B?B
�B~�B�3BR�B�B!��B-�B"23B�B�BpB�xB#۪B�B"B|�B}	B3��B%�rB%�(B�nB�B6;B��B-�B۾B(_vB��A��NB
��B �B�MBչB�B$�B�IB*�B�B/B�B�B��B�B]�B�B�=B�#B	x�BAXB?vB=�B	?�B�-B�B?�B��B�(B3�B��B8:B�B��B��BA�B��B!^�Bo�B"@tB 0�BKB?�B�B#�VB��B��BW�B��B3ªB%�FB%�!B��BńBFMB�DB-?�BÙB(��BA�A�~AB
C�B<�B��B,�B?�BD\B��B1�B��B4B�KB�EB��B@�BA�B	��A��A��nA�މA�εC�|@ASyA�e�C�S�A�܍B ��AsH�A]��A^+�A�kA���A�Y�A��A��A=��A���@�e�A��@�aA�ąA�y�A_+!AM�GA?�Ag�PA�BA��A��+AX�:@��A�B��@~>A4�
AAb�AKA6�@V�@��kA� 7A�%?A��aC���A�;�A8h0A�^XA�E@?	�X@��C���@���@�.c@J�}@�NC�� B	��A��A�uA��LA��C�o�AR��A��C�N�A��B �AsMA]LAb�A���A�uqA�C�A��9A��A>AAҔ@� A�d�@��A�~�A���A_�AN��A>��AgMeA�~�A��)A��vAX_8@�1�A bBsu@��A4�>AA��A�yA7$@Sz�@���A�OA�tA�{�C��A��A:ڰA���AƘ(?��@�,�C��@���@��A@Pe@�C���      
                  �      H   
               7      $            $   E            !         )      $               2                         *      	   �         )      H            &      B                           1      I            %      9                     ;         #   #         '                     1                                                !            !                                       !            %      '                     )         #                                 #                                                            !         N\N�1GN�ZO	��O*gOW��N VOI��NNTO�K�N�k�O�N��O�'On�P�wO@�O,*�Ny��N9��O ��O@�P �N���NL�MO�C�O���O�N�O��N{�MO��O��M���O!�	NS��OΔ+N�D�NQ�N��OZ9pOmzN�nORi�O]IOF7�NךOO	�Nw�O��O���N8��O�<N³@O P�N4��O��N�8KO�i�M�-k  �  �  �  �  �  �  U  �  w  �  �  �  �  �  �  �  [  N  e    
  n  �  �  �  L  �  F  
  �    F  �  u  u  �  b  �  �  �  �  O  �  /  �  4  �  �  �  �  �  �  
�  
  U  �      	�  ����j�49X��o�o�o���
=Ƨ�o=�w:�o;D��;�`B<t�<49X<���<49X<�1<�o<e`B<��
<ě�<�<�o<�/<��
<ě�<ě�<�=t�<�/<�h<�h<�h<��<��=P�`=C�=��=t�=�w=�w=49X=H�9=�+=aG�=Y�=���=e`B=m�h=�7L=�+=���=���=�{=�E�=�E�=�j=�/>%����������������������������������������`UXaantz�������zna``UW[egt|���������tg[U�����

��������wornqqsz����������zw��������������������LJKNO[gtx���}tkg[QNL#-/172/#����)9?@=5) �����������������������CCNO[hnt�����th[QOCC��������������������)5BIIRPB<;71)���	)*,*&"�����������������������������������������������������������������������������������),,*)#/<CHJPMH=<//(%#������������������������������������������������������������������������������/31-68Qgt����uj[B5/�����$(-01/)#��<:9BOP[[^_``a^[OLEB<��������

����

#/1<?AA</#
����������������������������������
#/0442/(#
��~���������~~~~~~~~~~ #%0<IJKIEEC<10#!#*054<20#!!!!!!!!	%)5BGMRSO@5)VX[^chtx���ytjh[VVVV56<HMU]YUHC<55555555-)/<>?><8/----------������	�����������������������������^bny{����{nb^^^^^^^^��������
�������4126;>HTV\`_ZTLHA;44oknt�������������{to�����������������������������


�����FHUXakkeaWUOKHFFFFFF����������������������������	�����������������������������������������������ABCLO[hnoih_[OIBAAAA������

��������)*+,)!&)6BOY\WUNB>3,)�����
����
	)6;@BCA<6)���������������������$�0�=�>�=�4�0�$�����$�$�$�$�$�$�$�$�G�S�`�l�y�{�y�l�f�`�S�G�<�E�G�G�G�G�G�G������������������������������������������������������������������������������������'�#��������������������������E�E�E�E�E�FFFFE�E�E�E�E�E�E�E�E�E�E��׾ؾ����׾׾Ӿʾʾʾоվ׾׾׾׾׾��B�O�[�f�h�r�s�l�h�[�O�B�6�-�)�'�+�6�;�BD�EEEEEED�D�D�D�D�D�D�D�D�D�D�D�D��0�<�I�Y�`�_�U�@�0�#��
�����������#�0�*�6�C�O�T�V�W�O�C�6�-�*�(�(�*�*�*�*�*�*������������������������������������������"�.�8�1�.�%�"���	����������	�
������"�;�G�X�`�e�`�T�;�.��	����׾оӾ����������������������������������������������	�� �&�&��	���������������������(�5�A�N�X�Z�g�n�g�^�Z�J�A�5�(�����(�(�5�A�N�R�Z�]�Z�Z�N�A�5�(�$������(������������������¿²§²²¿���������˾Z�Z�b�]�Z�M�A�=�A�A�M�X�Z�Z�Z�Z�Z�Z�Z�Z�����������������������������������������û˻û������������|�z�|������������B�T�[�I�)������������ú�������뼱�����ʼʼʼʼ��������������������������/�<�>�B�=�<�/�$�#� �#�*�/�/�/�/�/�/�/�/�ѿݿ�����(�0�;�;�5�(������߿ɿſѾ��	�.�G�R�Y�\�S�G�;�.�"��	�����������ʾ׾���׾վʾ����������������������A�M�Z�f�s�s�����x�s�f�Z�M�A�7�7�@�A�A�T�`�m�y���������|�y�b�T�G�;�0�!�%�.�;�T���(�1�(�%����	��	�����������*�6�C�O�\�h�_�O�C�*�������������������������ݿ׿ܿݿ���������������������������������������������������������������������q�f�d�r��ּ������������ּͼѼּּּּּּּ�Ƴ���������������������ƧƞƙƙƢƳ�-�:�B�F�S�X�U�S�G�F�F�:�6�-�+�$�-�-�-�-��(�*�(�(�&���������������f�s�~�w�s�f�Z�P�Z�e�f�f�f�f�f�f�f�f�f�f�y�������������y�l�`�Z�S�Q�R�R�S�`�l�u�y��(�4�A�E�I�B�A�4�,�(��������������������������������������������ʼռּ޼����ּʼ���������������������������
���
������������������������"�/�3�5�7�/�-�"��	�������������������������������������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D����������������s�q�s�u���������������������(�4�A�M�S�Y�W�M�A�4�*�(�����������
��"�%��
������������¿½¿�������n�z�{�z�y�r�n�a�W�X�a�k�n�n�n�n�n�n�n�n�ùϹܹ������������Ϲ����������üf�n�r���������r�f�c�Y�P�O�Y�b�f�f�f�fE�E�E�E�E�E�E�E�E�E�EuEqElEjEuE{E�E�E�E���������������������������������������������� �-�5�:�4�'������ܻлû��Ż���������������ֺɺƺɺֺֺ��e�r���������������������~�r�e�Y�R�U�_�eE7ECEPEREQEPECEBE7E6E7E7E7E7E7E7E7E7E7E7 K d Q 6 U F b  A = ( P D ^ O > r  r ; 0 ; 4 d B b ' _ 6 E T 7 5 _ 7 8 A T Y u - . � H ' : ;  W Q = W 1 4  M ` \ ( O  d  �  2  :  u  �  i  �  :  �  �  _  �  *  �  �  g  m  �  b    �  �  �  h    o  �  "  +  �  u  2  E  c  i  �  �  x    �  B  �  �  S  �    )  �  \    f  )  �    c  q  �      @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  �  �  �  �  �  �  �  �  �  �  u  g  W  G  @  C  G  9  #    �  �  �  �  �  �  t  b  O  <  )     $  *  3  >  N  b  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  e  J  '  �  �  �     I  k  �  �  �  �  �  �  �  �  �  �  m  L  '  �  �  G  �  y  �  �  �  �    z  s  g  X  F  .    �  �  �  �  �  k  D  �  �  �  �  f  -  �  �  0  0  -  ,      �  �  �  [  5    U  M  E  =  5  -  %           �   �   �   �   �   �   �   �   �  	  
�  �  \  �  �    �  �  (  i  �  `  �  G  R    
;  �  Q  w  l  `  U  I  >  3  '        �  �  �  �  �  �  �  �  �  �  �  �  �    P  v    {  �  �  �  �  y  a  ?  �  q  �  o  �  �  �  �  �  �  �  �  �  �  �  �  �  �  g  J  $  �  �  �  �  �  �  �  �  �  �  �  y  l  _  P  @  .      �  �  �  �  �  �  �  �  �  �  �  �  �  x  m  a  S  B  2  "   �   �   �   �  �  �  �  t  W  1    �  �  y  C    �  �  �  �  ]  (  �  �  �  �  �  }  n  [  H  3      �  �  �  �  [    �  �  a  <  I  x  �  �  �  �  �  �  �  �  z  S  '  �  �  T  �  y  �  4  [  L  =  -  (  %  "        �  �  �  �  �  n  K      �   �  �    *  =  G  L  L  @  '    �  �  x  4  �  �    �  �  �  B  A  G  P  ^  h  ]  9    �  �  r  7  �  �  .  �  s    �                    �  �  �  �  �  �  �  �  �  �  �  �  �  �    
       �  �  �  �  U    �  �    �  �      %  >  Q  b  l  n  d  P  1    �  �  @  �  �  [  �  o  �  d  .  @  E  V  �  �  m  R  1    �  �  P    �  k    T  ]  �  �  �  �  �  �    q  d  X  L  @  4  '      �  �  �  �  a  �  �  �  $  5  k  �  �  �  �  �  Z    �  z  &  �  &  �  �  L  @  '    '      �  �  �      ;  S  T  @    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    M    �  �  Y    �    "  :  D  D  =  1  "    �  �  �  d    �  r    �  q  (  �      	    
      �  �  �  �  g  4    �  �  Z  �  ^  �  �  �  �  �  �  �  �  �  �  w  '  �  n    �  3  �  G  �          �  �  �  �  �  �  �  �  �  �  �  }  d  J  0    7  F  >  1  .  :  :  -    �  �  �  l  /  �  �  :  �  +     �  �  �  �  �  �  �  �  �  v  e  U  G  :  4  0  &      �  u  p  k  g  b  ]  X  S  O  J  E  @  ;  5  0  +  &  !      u  n  g  \  L  8  #      %  -  &    �  �  �  _  $  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  *  D  K  K  N  Y  a  _  N  6    �  �  �  �  �  1  �  5  o  �  �  �  t  d  T  C  4  %      �  �  �  �  �  u  Z  ?  #  n  q  t  x  �  �  �  �  }  s  h  ]  N  @  -    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      &  6  F  U  e  u  �  �  �  �  �  r  _  G  )    �  �  �  \    �  o     �   L  O  <  -    	  �  �  �  �  �  m  K  '     �  �  o  ,  �  �  �  �  �  �  �  �  �  �  �  o  V  :      �  �  �  �  �    /  +  #    �  �  �  �  J    �  �  I    �  `  �  -  �  �  �  �    P  u  �  �  �  �  r  W  1  �  �  G  �  ?  p  _    +  1  3  2  .  &    	  �  �  �  �  f  @    �  �  �  S    �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  g  Y  L  @  3  �  g  �  p  �  W  �  �  �  �  e    �  "  4  �  q  �  	�  �  �  �  �  �  �  �  �  �  x  m  a  U  I  >  2  '  &  '  )  *  �  �  m  Z  J  L  B  $    �  �  o  &  �  �  *  �  i    �  !    8  �  �  �  c  <    �  �  u  .  �  �  R  �  =  �   �  
�  
�  8  �    u  �  <  �  �  �  ,  b  �  �  =  g  �  �  �  
�  
�  
�  
�  
�  
�  
�  
�  
�  
u  
@  	�  	�  	,  z  �  �  �  �    
       �  �  �  �  �  �  �  �  c  F  )    �  �  �  �  �  U  G  8  %    �  �  �  r  ?  
  �  �  $  �  K  �  H  �  �  �  �  t  b  N  :  "    �  �  �  �  u  R    �  �  4  �  �    �  �  �  |  G    �  �  L    �  l    �  b  �  G  �  �      �  �  �  �  |  \  9    �  �  �  K  �  �  W     �  7  	�  	�  	�  	�  	�  	�  	�  	�  	�  	�  	T  	  �  F  �     >    �  �    �  �  �  �  �  �  �  Z  .  �  �  t  6  �  �  �  S    �
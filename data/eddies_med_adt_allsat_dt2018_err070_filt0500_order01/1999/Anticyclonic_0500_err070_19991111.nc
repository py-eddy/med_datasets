CDF       
      obs    7   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�E����      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N!z   max       P�&�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �T��   max       =���      �  d   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�\(�   max       @E�z�G�     �   @   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @vh(�\     �  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @N            p  1p   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @��           �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �t�   max       >�v�      �  2�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       B B   max       B-�O      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       B H,   max       B-�:      �  4t   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >��K   max       C�      �  5P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��   max       C�"      �  6,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         ]      �  7   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9      �  7�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          +      �  8�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N��   max       P$�      �  9�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���ߤ@   max       ?��W���'      �  :x   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �T��   max       >?|�      �  ;T   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�\(�   max       @E�z�G�     �  <0   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @vg��Q�     �  D�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @          max       @L@           p  M`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�U�          �  M�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         G#   max         G#      �  N�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��S&�   max       ?��W���'     0  O�               -            	                  ,        \                        	   	   %      +   ;   3   
      	         g      3   
      D   	   '      :         E      �      	   N}��N�G�N�+�N>�O��MN��EN.�~NqH7Nz�COhoN�U�N�:�O�̚O��P�|O0��N�f9P�&�O�� O��Oy�O&%dN��N]��N˦HO	�bN��.O��O��PJ�O���O��N���N�,@N�^HN��N�'�P~��O[QdO��oN��MNB�P)^�N���O��SO��O�r?N	��N���O�M�N!zO�;�N>_�O�fO����T���49X�o:�o;��
;��
;ě�<o<o<t�<#�
<e`B<�o<�C�<�t�<���<�1<ě�<���<���<�`B=o=+=+=C�=\)=\)=t�=t�=�P=��=��=#�
=#�
=,1=0 �=0 �=8Q�=8Q�=@�=@�=P�`=aG�=e`B=m�h=m�h=q��=u=��=��=��-=���=�-=��=�����������������������/-,//8<HQOLJH?</////ECKNV[gnstuttg[NEEEE�����������������������
#<HMX^XUH</#�����	��������qtx�������~tqqqqqqqqwutz�������zwwwwwwww���������������������� ��������������%��������������������������[Z\acht��������tha`[XW[^chmt������ytha[X��������

�������HBHUanz�������zngYUH��������������������(5Ngt������tgN5%��������

�������#<EOPRUVUI<0+#"�������		�����)6>DEGDB64.)"��������������������]]hlt����zth]]]]]]]]mmxz������������zmmm��')555;5))5;52)sv|��������������~us�������

����������������������������{��������������������
/<KTSNLH</#
��ffhjt�������thffffff��������������������noqwz������znnnnnnnn���
�����������)5BNVPNB95))!������)7GA)����������������pos{��������������tp�����


��������������������������������������),&	�������{������������������@@FN[go{���}|tm[NDB@��������������������������������������������������������������������


�����RUanz��������znf^YUR�������������������������
 ������������������������������������������	������"�/�;�=�D�>�;�5�/�"���������D�D�EEEEE"E(EEED�D�D�D�D�D�D�D�D��b�n�{�~ŇŏōŇ�{�w�n�b�b�W�V�\�b�b�b�b²¿��������¿²¬ª²²²²²²²²²²��������������������������������4�@�F�D�B�@�9�4�'������'�0�4�4�4�4�����ûͻŻû��������������������������������������������������������������������l�x�������������������x�l�h�l�l�l�l�l�l�A�M�Z�^�h�s�y����s�f�M�A�:�7�2�0�1�7�A�����������������ּռӼռռּۼ����#�/�<�?�G�C�<�/�%�#�"� �#�#�#�#�#�#�#�#�A�M�Z�f�s�w�z�t�Z�Q�M�A��	������4�A�-�:�A�F�S�[�_�e�c�_�F�:�-�*�!���!�&�-�A�M�T�f�s�}����~�s�f�M��������(�A���������������������������������������˻������#�������������������)�6�>�E�E�A�4�������äßâø����������ûƻû����������x�r�`�a�r�x�������������������r�f�_�M�G�R�Y�V�Y�f�r�|����
��#�'�/�;�C�<�/�#��
�����������������ʾ׾�����׾ʾ����������������������y�������������������������{�y�y�y�y�y�y�L�Y�b�e�k�h�e�Y�T�L�I�G�L�L�L�L�L�L�L�LŠšŭůŹž������žŹŸŠŚŔŔőŔŔŠ�M�Z�f�j�k�j�f�d�Z�M�L�A�<�:�?�A�A�K�M�M�������������������������������������������	���!�&�%�-�"��	�������������������;�H�T�V�^�a�e�e�a�[�T�H�=�;�9�6�-�/�2�;�g���������������������s�(��������C�gù������������������ùìàÐÊÆÉÓàù�m�y���������������y�m�`�T�I�<�5�3�H�`�m�r���������������r�p�n�g�n�r�r�r�r�r�r�r���������������y�r�p�n�n�r�r�r�r�r�r�-�:�F�S�W�W�S�F�:�-�$�$�-�-�-�-�-�-�-�-�����������������������������������������m�y�v�w�t�w�m�`�^�X�T�R�T�T�`�g�m�m�m�m���������������������Y�F�I�D�'���3�e���.�;�G�T�`�h�p�m�j�`�T�G�;�.�����"�.�4�A�M�Z�f�s������s�f�Z�4�(�"����0�4�Z�f�s�x�w�s�l�f�Z�Y�M�M�M�V�Z�Z�Z�Z�Z�ZÇÓàçàÞÓÇÃÂÇÇÇÇÇÇÇÇÇÇ�)�5�N�l�y�g�[�B�)����������)���
�������
�����������������������(�A�K�R�R�N�A�5������������
����������ĿѿտѿƿĿ������������������������'�4�@�M�Y�m�j�M�'����ܻʻǻʻ߻��/�3�<�C�<�/�)�#��#�#�+�/�/�/�/�/�/�/�/E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E}E{ExE�E������� �����ܹϹù������������ùϹܹ�l�x�������������x�p�l�c�l�l�l�l�l�l�l�lD�D�D�D�D�D�D�D�D�D�D�D{DxDwD�D�D�D�D�Dƽ���������������������������������������Ƴ��������������ƼƳƧƜƚƗƕƖƚƧƪƳ�����������
��#�.�.�#��
��������ĵĿ�� W O ) & O S 2 3 _ I o [ ( 5 E x .  C T 6 U > Q ` S h < E k ' - @ P 3 n \ S / 4 - F ' h ( Y i T 0 5 c 2 j D f    �  �    S  x  �  O    �  �  \  �  �  M  �  �  �  ?    �  �    �  �    I  �  �  Y  �    "  �  �  �  4    @  �    �  V  �  �  n  :  �  9      ?  ^  n  I  K�t�%   <D��;��
=H�9<u<���<e`B<�C�=�P<�/<���=0 �=t�=y�#<�/<���>�v�=L��=��=@�=0 �=8Q�=aG�=49X=49X=0 �=��=P�`=��
=ě�=�E�=L��=]/=L��=8Q�=D��>�P=�C�=ȴ9=e`B=�+=��m=�o=Ƨ�=�\)=�h=�C�=ě�>hs=���>T��=�^5=��`>%B��B�	B�B�[BaGB�GBøB��B�.B�B_�B�"B�[B�B#�B��B!�B�,B#B&$BbB��B!LBB BB,$BH�B��B/�B�zB6B�hBi�B �%Bo�B6fBBGB?�B{�B�-B@B"�'B�<B[�B	�BDB^�B?�B�B�B,�KB EB-�OB(�B�5B�_B�MB�4B 7�B��B��B��B��B�B?�BJ�B��B��B@�B#>CB�B �DB��B#=�B%��B+�BE'B!>mB?�B H,B5�B}B�TBA�B@dB��BA�BApB �gBf-B�`Bo�BAB��B��B< B"C:B�B]B�4B?�B@YB?�B>`B@'B,��BB-�:B�B��A��C�X�A�FA�pMAҬF@̥{@�-A��U@�NuA=� AЅA�A:��@�WA9B�A���@�|�A��,@��@��oA��AO�	A�?��aA�WA=�A���A��;A���A�|�A�~�Aj�-@�p@��@},�Ar�Aj٧?�X�Ad�A<��A@�)A�e]A��A��]A���AuGu@�&�A�q�C�>��K@�\�C���A"�TB?A檙A�~�C�e�A��0A���Aҁo@ͺ�@�)�A���@�?�A>A�AtA<k@���A8��A�rC@��Aҁ�@�N�@�:%A���AP:�A�?���A�R�A>��A�8YA���A���A��cA͈Aj��@��3@��@{�SAr�`Ai��?�KAc�A<��A@�WAʃ�A���A�r�A�~�As��@�~�A�o�C�">��@���C��`A"r�BjpA�               -            
                  ,   	     ]                        
   	   &      ,   ;   4         	         h      4   
      E   	   (      :   	      F      �      	                                          #      )         5                              !      5      !                  9      %         -            )                                                                     #                                       !      +                                       '                                    N}��N�G�N�+�N��O2wN��ENghNqH7Nz�CO<�vN��TNf��O���N�ɫO�77N�bN�f9O��O�� O��O0�^O&%dN��NF`N˦HO	�bN��.O��O��P$�O���O�{	NoBN�)�N�^HN��N�'�O��OM�O�H�N��MNB�P��N�?*O��SO��O|CN	��N�7O�M�N!zO 3N>_�O�fO���    M  �  �  �  ;  �  �  Z  �  �    %  �  ~  :  =  C  ^  .  �  V  )  �  �  �  M  �  H  �  
  �    �  �    �  
.  �    �  �    ?  �  �  o  %  �  �  ^  �    E  �T���49X�o;o<��
;��
;�`B<o<o<T��<D��<u<�1<�1<�`B<�1<�1>?|�<���<���=o=o=+=�P=C�=\)=\)=t�=t�=,1=@�=T��=,1='�=,1=0 �=0 �=�E�=<j=u=@�=P�`=�o=ix�=m�h=m�h=�{=u=���=��=��->�=�-=��=�����������������������/-,//8<HQOLJH?</////ECKNV[gnstuttg[NEEEE�������������������� #/<@HMSN@</,#����	��������rty��������trrrrrrrrwutz�������zwwwwwwww����������������������������������������������������������������fe`_djt����������thfZY[bfhrtz�����}tih[Z�������
	�������JIUanz}����zpnla^UJJ��������������������24;BNgt|����tg[NB62��������

�������#<EOPRUVUI<0+#"������	�������)6>DEGDB64.)"��������������������dbhqt|���wthddddddddmmxz������������zmmm��')555;5))5;52)sv|��������������~us�������

��������������������������������������������������
#/=EGD</#
�gghnt�����thgggggggg��������������������noqwz������znnnnnnnn���
�����������)5BNVPNB95))!�������� ������������������~{}����������������~�����


������������������������������������!&!������}�������������}}}}}}@@FN[go{���}|tm[NDB@��������������������������������������������������������������������


������RUanz��������znf^YUR����������������������������

	����������������������������������������	������"�/�;�=�D�>�;�5�/�"���������D�D�EEEEE"E(EEED�D�D�D�D�D�D�D�D��b�n�{�~ŇŏōŇ�{�w�n�b�b�W�V�\�b�b�b�b²¿��������¿²¯¯²²²²²²²²²²��������
�����������������������4�@�F�D�B�@�9�4�'������'�0�4�4�4�4�����û̻Ļû��������������������������������������������������������������������l�x�������������������x�l�h�l�l�l�l�l�l�A�M�Z�\�f�s�v�{�s�Z�M�A�>�6�4�3�4�5�=�A�����������������ּռ׼ּּּܼ����#�/�<�=�E�B�<�/�&�#�#�"�#�#�#�#�#�#�#�#�(�4�A�M�Z�f�s�q�m�f�Z�M�A�4������(�-�:�=�F�S�V�_�a�_�[�S�F�:�2�-�#�!�!�,�-�(�4�A�U�e�p�u�s�k�Z�A�(����������(���������������������������������������׻������#��������������������"�'�(�$�����������������������������ûƻû����������x�r�`�a�r�x�������������������r�f�_�M�G�R�Y�V�Y�f�r�|��
��"�)�/�4�/�.�#��
����������������
���ʾ׾�����׾ʾ����������������������y�������������������������{�y�y�y�y�y�y�L�Y�^�e�g�e�^�Y�X�L�K�J�L�L�L�L�L�L�L�LŠšŭůŹž������žŹŸŠŚŔŔőŔŔŠ�M�Z�f�j�k�j�f�d�Z�M�L�A�<�:�?�A�A�K�M�M�������������������������������������������	���!�&�%�-�"��	�������������������;�H�T�V�^�a�e�e�a�[�T�H�=�;�9�6�-�/�2�;�g�����������������s�N�(��������N�gÓàìù����������������ùëà×ÐËÑÓ�`�m�y���������������y�`�T�I�B�A�G�H�R�`�r�������������r�p�j�q�r�r�r�r�r�r�r�r�r���������������z�r�q�o�p�r�r�r�r�r�r�-�:�F�S�W�W�S�F�:�-�$�$�-�-�-�-�-�-�-�-�����������������������������������������m�y�v�w�t�w�m�`�^�X�T�R�T�T�`�g�m�m�m�m�e�~�������������~�r�e�Y�L�@�5�*�,�9�L�e�.�;�G�T�`�f�n�m�i�`�]�T�G�.�����"�.�4�A�M�Z�^�n�u�y�u�f�Z�M�A�4�+�"�!�"�'�4�Z�f�s�x�w�s�l�f�Z�Y�M�M�M�V�Z�Z�Z�Z�Z�ZÇÓàçàÞÓÇÃÂÇÇÇÇÇÇÇÇÇÇ�)�5�B�N�d�o�t�p�g�[�B�)�����������)�
�������
������������
�
�
�
�
�
��(�A�K�R�R�N�A�5������������
����������ĿѿտѿƿĿ�������������������������'�(�/�'� ���������������/�3�<�C�<�/�)�#��#�#�+�/�/�/�/�/�/�/�/E�E�E�E�E�E�E�E�E�E�E�E�E�E�EE~E�E�E�E������� �����ܹϹù������������ùϹܹ�l�x�������������x�p�l�c�l�l�l�l�l�l�l�lD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�����������������������������������������Ƴ��������������ƼƳƧƜƚƗƕƖƚƧƪƳ�����������
��#�.�.�#��
��������ĵĿ�� W O ) ) + S 5 3 _ G j K  4 @ ^ .  C T ) U > N ` S h < E `   . < O 3 n \ 6 / * - F ( q ( Y : T & 5 c  j D f    �  �      J  �  :    �  �    �  !    �  %  �  �    �  �    �  A    I  �  �  Y  �  �  ;  �  �  �  4    �  �  N  �  V  b  �  n  :  [  9  �    ?  =  n  I  K  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#  G#    ~  }  |  {  z  y  y  y  y  r  e  X  K  >  (     �   �   �  M  J  C  1      �  �  �  �  c  5    �  �  e  "  �  �  O  �  �  �  �  �  �  �  ~  w  m  `  M  1    �  �  X    �  �  s  v  z  }  �  �  �  �  �  �  �  �  �  �  �  �  �  _  .  �  �  �  �  -  W  s  �  z  `  2  �  �    �  {    l  �  h  +  ;  %       �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  h    �  �  �  �  x  `  F  +    �  �  �  {  S  *  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  g  d  z  �  �  �  �  �  Z  D  .    �  �  �  �  l  7  �  �  g  $  �  �  �  �  �  u  �  �  �  �  �  �  �  �  �  �  d  8    �  }    �  *  �  �  �  �  �  �  �  �  �  �  �  m  G    �  �  �  I    �  A  S  �        "            �  �  �  �  �  �  |  *  �  �  �  �    "  #                 �  �  �  �  T  �  �  %  �  �  �  �  �  �  �  �  �  �  �  u  [  5  �  �  n     �  d  �  !  O  n  |  {  l  S  3  
  �  �  o  .  �  �  3  �  �   �        !  /  8  ,  !           7  =  D  B  >  6  %    =  1  %      
      �  �  �  �  �  �  �  �  �  d  8       �  �  �  �  �  �  �    C    �  �  �  �  �    �  g    ^  X  Y  U  N  @  5  '       �  �  �  �  �  o  =      �  .             �  �  �  �  �  �  �  �  _  =    �  5   �  Q  Z  a  `  �  v  b  H  ,    �  �  �  z  T    �  �  V    V  P  I  A  5  &    �  �  �  �  �  ^  8    �  �  �  J   �  )  $      �  �  �  �  �  �  ~  e  F  "  �  �  �  p  ?    b  �  �  �  �  �  �  `    �  �  9  �  �  7  �  x    �  ?  �  �  p  [  F  )  
  �  �  �  �  w  T  -  �  �  �  P  �  U  �  �  �  �  �  �  �  �  �  �  w  e  L  1    �  �  �  X    M  5      �  �  �  �  �  j  M  /    �  �  �  �  �  �  �  �  �  s  b  \  R  B  0      �  �  �  B  �  ~    �    �  H  :  )      �  �  �  �  �  �  n  M  *    �  �  �  q     d  �  �  �  �  ~  U  !  �  �  Z    �  |  B    �  �  �  �  	�  	�  	�  
  

  	�  	�  	�  	X  	  �  _    �  L  �  X  �  �  4    >  x  �  �  �  �  �  �  ~  Q    �  T  �    m  �    o  �  �        �  �  �  �  �  �  �  b  A    �  �  �  �  f  �  �  �  �  �  �  �  v  d  Q  =  '    �  �  N  �  y  �  �  �  �  q  _  L  :  *      �  �  �  �  �  �  z  d  0  �  i            �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  i  ]  P  D  9  0  '      �  �  �  	�  	�  	�  	�  	`  	  	�  
  
.  
  	�  	�  	s  	  s  �  �  n    t  �  �  �  �  �  �  ~  p  ]  >    �  �  m  /  �  �  5  �  1  %  Y  l  w  ~    w  e  H  %  �  �  �  ?  �  k  �  
    �  �  �  t  Y  =    �  �  �  �  Q  !  �  �  �  �  n  S  E  6  �  �  �  �  �  m  N  *    �  �  v  A    �  y  4  �  �  k  �  �             �  �  �  L    �  M  �  :  �  �    �  =  >  >  ;  4  +      �  �  �  q  J     �  �  �  a  0     �  �  �  l  M  &  �  �  �  c  &  �  �  C  �  �    �  }  V  �  �  �  �  �  �  o  R  5    �  �  �  ~  P  9  '    �  �  �  �  �  �    1  0  8  f  l  R  )  �  �  2  �      �  #  %    	          �  �  �  �  �  �  �  �    2  X    �  �  �  �  �  �  x  Q  !  �  �  �  G    �  h    �  1  �  )  �  �  �  �  �  }  B  �  �  -  
�  
!  	�  �  E  �  �  }  �  �  ^  j  v  �  s  _  K  1    �  �  �  �  �  f  G  (     �   �  o  �    @    �  �  �  �  �  �  <  �    .    �  T  �  
[    �  �  �  �  �  �  �  �  �  �  |  k  Z  I  +    �  �  �  E  C  A  :  1  %      �  �  �  �  �    l  Y  G  3      �  �  �  �  �  �  m  M  (    �  �  9  �  "  �  �  W  �  x
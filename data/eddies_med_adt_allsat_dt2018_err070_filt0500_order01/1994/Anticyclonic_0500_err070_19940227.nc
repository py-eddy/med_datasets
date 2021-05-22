CDF       
      obs    4   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�"��`A�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�8�   max       P�@�      �  |   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��C�   max       =�-      �  L   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�ffffg   max       @E�33333            effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��(�`    max       @vp�����        (<   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @M�           h  0\   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ɖ        max       @�`          �  0�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �49X   max       >/�      �  1�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B,�=      �  2d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��k   max       B,��      �  34   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?ٯ   max       C�i�      �  4   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?+m   max       C�g�      �  4�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          j      �  5�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9      �  6t   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3      �  7D   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�8�   max       P�@�      �  8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��vȴ9X   max       ?��G�z�      �  8�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �u   max       =���      �  9�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @E��Q�        :�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��(�`    max       @vpz�G�        B�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @M�           h  J�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ɖ        max       @�`          �  K,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?    max         ?       �  K�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��Q�`   max       ?�ߤ?��     @  L�               #                        +   *         	         	   $      /            8                                                )                "   
      j   %N���O,\CN���N�xO���Op��O�O��N= O��5N�PN :=O�.�O���NP��N�EN�F�OU��O�!�O�OܱN&irPh�CN�7N1gO��P�@�N�NީDN9�O��}O"O�Q'Ne�NNZ6[N��FNA��OT�OX��N{�-M�8�Nnc$Pw�Nf�O[�O1O���O��N['�O�&�O�B
O��s��C��u�u�u�t��o:�o;ě�<#�
<#�
<#�
<T��<e`B<e`B<e`B<e`B<e`B<u<u<�o<�o<�o<�C�<�C�<���<��
<��
<�1<�9X<ě�<ě�<ě�<���<���=o=+=+=+=C�=\)=\)=t�=�P=�P=�w='�=@�=P�`=ix�=�o=�+=�-hktu�������thhhhhhhhd`bcgt��������ytsjgdebgpt�������tgeeeeee��������������������glx��������������mgg��������������������WQPS[hmt�������zth[W22468>BOTX[[ZVOKB962�����������������������������������������������������������������������������������
/<RQK<# ��#/<HU^f[OQ\]U</#��������������������'$)6ABIB6)''''''''''������������������������������������������������������������������������������������)6BKOQOGB6)������������������������������<IB�������������������������/*02<>BA<0//////////��������
	����������5BNVYVNB)���5;<HUWUTH<5555555555)*0.-)

#./0/,*##




79<ACHUanx{||xqnaU<76BMO[hostrhh[POFB;66���������������������{������������������)*-3)'$)565//)��������������������������������������������������������������������������������#')#����
 ��������������%)'������������������������"/5;=@@;8/"������������������#/7BCEB8)#�������������������^VV[adknorqnla^^^^^^�||��������������������������	
�����zlgihdmz�����������z�L�Y�e�e�p�m�e�Y�M�L�A�@�L�L�L�L�L�L�L�L�zÇàìõóöìàÓÇ�z�n�d�a�_�a�n�r�z����������������������û��ÿ�����������ź����������������������������������������R�[�h�l�p�s�o�h�[�O�B�6�3�*�'�&�-�B�L�R���5�A�J�W�[�Z�N�A�5�(��������
��������������������������������������������������������������������y�n�l�a�l�q�����ĿͿǿĿ��������������������������������	�"�1�;�H�T�V�]�=�/�"����������������Z�c�^�]�Z�M�L�A�@�A�M�U�Z�Z�Z�Z�Z�Z�Z�Z²¿����¿²¦¦±²²²²²²²²²²�.�;�I�T�`�m�����������m�T�G�?�:�;�.��.������������������ùìçù�������������������z�w�u�y�z���������������������������������������������� ���������������������������������������������������ý�ź��)�)������ܹϹù������ù̹ܹ����;�>�H�Q�T�X�U�M�H�;�7�/�(�"��� �/�1�;�	��/�2�6�4�0�$��	�������������������	��������������������������m�����ѿ��Ŀ������y�`�;�4��.�3�;�G�m�/�<�H�T�R�H�A�<�/�#�#�!�#�'�/�/�/�/�/�/���������������w�|�����������h�tčĚĦĳļľĽĳĦĚā�t�T�H�N�[�`�h�#�<�Q�gŊŏň�{�t�b�#�
��������������#²¿¿����¿¶²©©²²²²²²²²²²�h�u�vƁƍƎƚƤƞƚƎƁ�u�q�k�h�f�h�h�h���(�)�0�5�7�5�4�(�����������N�Z�_�s���������������s�Z�A�5�2�3�6�A�N���������׾ʾ������������ʾϾ׾��нݽ��������ݽĽ�����������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eͼ�'�4�@�D�A�@�4�'�������������"�/�1�/�,�"�����	���������	����f�s�������s�f�d�a�f�f�f�f�f�f�f�f�f�fÓàèìùý��ùøìàÓÓÇÂÁÄÇËÓ�ܻ����� ��������ܻлû������������
�������
�����������
�
�
�
�
�
�����������������������������������������	��"�#�*�"���	� �������	�	�	�	�	�	�����������������������������������������ʼμּ׼ּӼʼ��������üʼʼʼʼʼʼʼ��"�/�;�H�T�V�]�V�H�?�;�/�"��	�����	��"���������
��
�����������������������-�F�S�Z�U�F�:�!������������ �
�!�-�y�������������½����������y�t�n�l�j�l�y���ûлܻ߻ܻлȻû����������������������������ռ�������ּʼ���������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�DwDmDpDuD{D��zÇÓìù����������������ùìàÚÈ�z�z D J ? 7 < , H K n 7 Q m > G w \ J ? l 0 = g Y : = 3 2 X L d - 8 M G - O 5   L q : H - A ^ 8 l - d P  0  �  �  �  �  o  �  a  i  b  u  =  -  �    �  E  �  �  I  a    b  
  �  !  �    Y    �  d  ^  �  �  o  �  ^  9  �  �    k  t  <  �  %  �    y  X  &  E�D���o�49X��o<���<T��<�C�<���<T��=\)<D��<��
=e`B=aG�<�t�<�o<�9X='�=��<ě�=P�`<��
=�%=o<�1=P�`=���=+<���<�/=L��=��=L��=�P=t�=�w=t�=aG�=]/=0 �=�P=#�
=��-='�=�+=Y�=���=�1=�+=�{>/�=��B��B	��B	�B!�B7�ByjB{Bm]B'.B^0B:<B/�B��B�XB��B��B!.�BB+�B^?B;B gaB�B�\B&�BX�B�jB�B�B��B�B&�B �B**B��B`DBmB!�B"B��B% �BƯB��B"�&A��Bu{B��B,�=Bz�B'�BۯB kLB��B	��B
B"6 Bj�B]`B�jB?�B �BB�B5XBA�B�B�7B>�B��B!>�B"�BA$BBdB;�B zzB�3B��B%�B�BBB=�BO<B�BL�BuUB!�B?�B�BEhB��B"=B?PB��B$�%B��BRbB"��A��kB�6B?�B,��B?'BN�BşB @�?�.NA��A�B@��A�%�A���@�b�A�=AwG�A�}�A>kA���Ai��A�V]A�b�A�)@�ZAЄ�?ٯA�^9A�N?RnEAi��A�?{@���A�)wA�JwA���B<A�I$A��AR�+A*J�C�i�@��A�zAC&A�^�@��-A��ALĠA\WA��=@��"A�<�A��@o��A��@�l�@��RC��/A�c�?�AɃA�{z@��A��/A�m�@�FsA�Ax�"A�~gA=��A��,Aj>�A�x%A��/AD�@��A�wn?+mA���A�W�?Q�LAi��A�	�@��8A�~A�$A���B�(A��A��AQ��A+uoC�g�@�N%A��sAC�+A�x~@���A��AL�6A\KZA���@���A�wA�@tA�@���@��dC�׀Aˡ�               $                        +   *         	         	   $      /             9                                       	         )            !   "   
      j   &                                       %   !               !      !      9            3                  !                              %            !                                                                        !                        3                                                %                           N8�N��BN���N�xOI�O5DO�O��N= OړN�PN :=O��N��NP��N�EN�F�N��IO�!�O�O��N&irO,�'N�@MN1gOn9P�@�N�NީDN9�O=�O)�OJ��N>-�NZ6[N��FNA��OT�O,��N{�-M�8�Nnc$Pw�Nf�O7�[O1Oe$rO$"�N['�O�&�O�OD{$  "  �  �  �     p  �  v  O  �    �    �  
  6  Z  �  �  Q  �  M  �  �  �  �  �  �      F  �  �  �  �  �  (  9    �  �  3    \    i  �  2  �  �  h  ��D���D���u�u;�o:�o:�o;ě�<#�
<��
<#�
<T��<ě�=C�<e`B<e`B<e`B<�/<u<�C�<�j<�o=8Q�<���<���<�h<��
<�1<�9X<ě�=+<���=C�<�/=o=+=+=+=t�=\)=\)=t�=�P=�P=,1='�=H�9=y�#=ix�=�o=���=ě�oot~�������toooooooogcddfgqt�������tmgggebgpt�������tgeeeeee��������������������y~�����������������y��������������������WQPS[hmt�������zth[W22468>BOTX[[ZVOKB962�������������������������������� �������������������������������������������������
#/@GLKH@<#
��**,/<HKQMH><5/******��������������������'$)6ABIB6)''''''''''������������������������������������������������������������������������������������)6BEJLLIB6)�����������������������������������������������������/*02<>BA<0//////////���������

	���������5BNVYVNB)���5;<HUWUTH<5555555555)*0.-)

#./0/,*##




FDFMSUalnrtuusnaUSHF=:BOQ[dhnrsqh[ROGB==����������������������������������������)*-3)'$)565//)��������������������������������������������������������������������������������#')#����
 ��������������%)'������������������������"+/3<>>;5/"���������������������".6?=6)���������������������^VV[adknorqnla^^^^^^�||��������������������������

	�����wqprtz|�����������zw�L�Y�`�e�h�e�[�Y�X�L�H�G�L�L�L�L�L�L�L�L�zÇÓàìîììëàÓÇ�z�n�j�e�n�x�z�z����������������������û��ÿ�����������ź����������������������������������������O�[�]�h�h�i�h�d�[�O�G�B�6�5�/�-�6�B�D�O��(�5�A�D�R�S�N�A�:�5�(�����
����������������������������������������������������������������������y�n�l�a�l�q�����ĿͿǿĿ��������������������������������	��"�,�/�:�7�/�"���	��������������Z�c�^�]�Z�M�L�A�@�A�M�U�Z�Z�Z�Z�Z�Z�Z�Z²¿����¿²¦¦±²²²²²²²²²²�m�y�����������y�m�`�T�O�G�E�A�D�M�T�`�m������������������������������������������������z�w�u�y�z���������������������������������������������� ����������������������������������������������������������Һ��)�)������ܹϹù������ù̹ܹ����;�=�H�O�T�S�I�H�;�/�+�"���"�"�/�6�;�;���	��"�)�0�.�(���	��������������������������������������������m�y����������y�m�`�T�H�G�F�G�G�N�T�`�m�/�<�H�Q�N�H�@�<�/�%�#�)�/�/�/�/�/�/�/�/���������������w�|�����������tāčĔĚĦīĳĲīĦĚčā�u�g�\�h�m�t�#�<�Q�gŊŏň�{�t�b�#�
��������������#²¿¿����¿¶²©©²²²²²²²²²²�h�u�vƁƍƎƚƤƞƚƎƁ�u�q�k�h�f�h�h�h���(�)�0�5�7�5�4�(�����������N�Z�g���������������s�g�Z�N�B�=�@�A�K�N�׾����������׾ʾ����������ʾѾ׾׽Ľнݽ�����������ݽнĽ�����������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eͼ�'�4�@�D�A�@�4�'�������������"�/�1�/�,�"�����	���������	����f�s�������s�f�d�a�f�f�f�f�f�f�f�f�f�fÓàèìùý��ùøìàÓÓÇÂÁÄÇËÓ���ûлܻ����
�������ܻۻлû»������
�������
�����������
�
�
�
�
�
�����������������������������������������	��"�#�*�"���	� �������	�	�	�	�	�	�����������������������������������������ʼμּ׼ּӼʼ��������üʼʼʼʼʼʼʼ��"�/�;�H�N�T�Z�T�R�H�;�/�"��	�����	��"���������
��
������������������������!�-�:�F�S�Y�T�D�:�!����������������������������������������{�y�u�r�y�|�����ûлܻ߻ܻлȻû����������������������������ռ�������ּʼ���������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{D{D}D�D�ÇÓàìù������������ùìàÓÎÇÁÅÇ > H ? 7 ) ) H K n  Q m 1  w \ J 5 l / 3 g . < = & 2 X L d 3 4 ? B - O 5   2 q : H - A T 8 [ & d P  ?  J  +  �  �  T  z  a  i  b  6  =  -    �  �  E  �  �  I  7  f  b  s  �  !  �    Y    �  �  3  �  b  o  �  ^  9  v  �    k  t  <  �  %  #  ^  y  X  @  �  ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   �  �  �  �  �      "  !        �  �  �  �  �  �  d  7  i  w  �  �  �  �  y  g  R  ;    �  �  �  h  "  �  �  =   �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  p  b  T  F  8  *  �  �  �  �  �  �  �  �  �  �  s  \  A    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  Y  &  �  {    �  �  [  g  m  p  o  k  c  Z  O  A  .    �  �  �  D    �  a   �  �  �  �  �  �  �  �  �  �  o  T  J  `  S  D  +    �  �  �  v  `  d  J  ;  *    �  �  �  i  5  �  �  O  �  +  �   �   �  O  I  C  =  7  0  *  #          �  �  �  �  �  �  �  �  �  "  Z  �  �  �  �  �  �  �  �  �  �  g  6  �  �  6  ~   �              �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    u  k  d  \  +  �  Y  �  �  8  �  �  �          �  �  �  c  "  �  �  /  �    l  �  <  �  &  �  �    V  |  �  �  �  �  �  �  T    �    c  �  4  
    �  �  �  �  �  �  �  �  �  �    k  X  G  6  %      6  2  -  (  $            
             �   �   �   �  Z  O  D  9  .  #      �  �  �  �  �  �  �  �  �  �  p  _  W  �  �    /  P  m  �  �  �  ~  k  R  4    �  �  �  %  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  V    �  >  �  N  O  Q  M  G  ?  0  !    �  �  �  �  �  �  �  o  [  C  ,  �  �  �  �  �  �  �  �  �  �  e  3  �  �  n  $  �  Z  �   �  M  L  K  I  H  F  C  ?  ;  7  1  *  #          �  �  �  �  �  �    4  C  M  _  o  �  �  �  �  �  z  B  �  b  �  �  ~  �  �  �  �  �  �  �  |  n  W  >     �  �  �  u  A    �  �  �  �  �  �  �  �  �    |  s  d  U  E  6  &       �   �  �  �  �  �  �  �  �  �  �  r  E    �  {  $  �  n    �    �  �  �  m  d  T  B  .      �  �  �  �  �  E  �  i  �  �  �  �  �  �  �  �  �  �  g  -  �  �  �  r  G    �  �  �  c    �  �  �  �  �  �  �  �  �  �  �    q  c  Y  P  H  ?  7        �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  o  e  �  �    #  2  =  D  E  A  0    �  �  �  S    �  X  �  �  �  �  �  �  �  �  �  �  �  |  c  H  %  �  �  �  v  O  '  �  l  t  y  �  �  �  �  �  �  |  f  J  (    �  �  �  �  u  j  ^  z  �  �  �  �  �  �  q  N  $  �  �  d    �  �  0   �   �  �  �  �  ~  w  o  c  W  K  ?  /      �  �  �  �  �  �  x  �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  @       �  �  (              �  �  �  �  �  �  �  �  �  �  �  �  �  �  9  7  "    �  �    J    �  �  r    �  �  �  �  j  =    �  �      �  �  �  �  �  �  �  �  k  @  	  �  �  D  �  �  �  �  �  }  k  W  ;    �  �  �  �  T  #  �  �  j  !   �   �  �  �  �  �  �  �  �  �  �  �  |  v  q  k  f  `  [  U  P  J  3  /  +  '  #         �  �  �  �  �  �  �  j  O  5     �    �  �  �  ^  �  Y  ]  ~  �  �  �  �  }  E     �  N  �  �  \  O  C  6  *      �  �  �  �  �  �  �  y  a  G  .     �  �          �  �  �  �  Y  ,     �  �  Y    �  �  /  �  i  _  U  J  D  >  5  )      �  �  �  �  n  R  :  &       �  �  �  �  �  �  �  \  ,  �  �  �  :  �  �  I  �  :  �  �  �        (  0  1  )    �  �  �  e    �  �  ,  �  $    �  �  �  �  |  `  D  &    �  �  �  x  C  �  �  ]    �  P  �  �  �  �  �  o  R  2    �  �  �  u  8    �  �  E  �  �  �    r  �  �  3  X  g  X  (  �  {    {  �  �  �  
�  �  %  r  P  >  <  �  �  �  �  t  X  /  �  �  G  �  (  �  �  �  d
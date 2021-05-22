CDF       
      obs    2   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�������      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       NG�   max       P\Q      �  t   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �+   max       =�x�      �  <   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��\)   max       @E��\)     �      effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�33333    max       @v�G�z�     �  '�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @P            d  /�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�R�          �  0   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���
   max       >H�9      �  0�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�>�   max       B-�      �  1�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�s�   max       B,��      �  2`   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?O�    max       C��}      �  3(   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?P�   max       C��
      �  3�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          e      �  4�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min          
   max          3      �  5�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min          
   max          /      �  6H   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       NG�   max       P��      �  7   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��j~��#   max       ?۟U�=�      �  7�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��`B   max       =�x�      �  8�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��\)   max       @E��\)     �  9h   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @v��\)     �  A8   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @P            d  I   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�          �  Il   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >�   max         >�      �  J4   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���҈�   max       ?۟U�=�     �  J�                        '   d   =      
      -   
                     1   "      0                     7   #                        -      	            )      X   	O'�N���NG�Nf��O�+�O��N0J	O��PG�PPgO���O�HO@�fPK��N��QN�Z�NG�6NDjO	�%N7�N��TO��pO�OY@�P\QOY��OZl�N��N�*�N��Og�1P��O�*"O'�O[�SN'"NR�OL��O+�0N��(O�n�O5��N�+�Nd�%O0M&O�O��dN�
mO]N"�~�+��`B�ě��D����o:�o;�o;ě�;ě�<49X<49X<D��<T��<e`B<e`B<�C�<�C�<�C�<�C�<�1<�9X<�9X<�9X<�j<�j<���<���<��<��=o=+=\)=\)=t�=�P=0 �=49X=8Q�=8Q�=D��=D��=ix�=m�h=m�h=��=��=Ƨ�=���=�;d=�x�fccegnz������zvtomjf����������������������������������������mot{�������tmmmmmmmm�������� ����������5BHMHPS]^[NB5)WU[^hqoih[WWWWWWWWWWUSUH#
������
/8HUbm}���������������kb�����������������5469BO[httomg[OB;75����������������������������������������'$)/<Hn��������zaH<'���������������������")2*)��������������������������COT[fhktttrh[OCCCCCC|vu���������������||;:<IUWZXUID<;;;;;;;;����
#')&#
�������/<H\a^TVULH<:1!'**04>HUnz���~naH</'������������������)BNY__\K5���"/58;HLHE>;/"��������������������"$&)5=>@>85)""""""����������06BO[ahhhh[OLB:60000),5;?=<;5)������)-,����������)6?HHB;)����� 
#04<920#

����������������������\\Zadknrnnma\\\\\\\\MHGNV[dghhge[NMMMMMM�������������������������������������zyz���������������zz�������%)������KHLKR]ht~����thb^[OK������������������)+)&,664)�����),`^`amz������zmhaa``gbabdhmz���������zmg����������������������������

�����",-$"�a�nÇàãàÓÏÇ�z�n�a�P�H�<�1�<�H�U�a�G�T�W�_�`�b�d�`�T�G�E�;�:�9�;�@�G�G�G�G��������������������������������������L�Y�c�e�i�i�e�Y�Q�L�E�D�L�L�L�L�L�L�L�L�����������������������s�o�s�s�o�u���������)�7�M�Q�O�B�)�����������������������������������������������������������l�`�G�?�4�.��	��;�T�`�y�����������u�l�O�[�h�s�r�[�O�B�A�6�)���������6�O�5�B�[�g�t�[�N���������������5���������ǽͽϽɽȽĽ��������x�o�o�y�������������� ����������������	��"�/�2�8�9�0�/�"��	���������������a�m�������������������z�p�n�w�l�P�H�N�a������������������������Z�a�f�j�i�f�b�Z�M�L�H�M�O�V�Z�Z�Z�Z�Z�Z�����	�	��	�����������������������������H�L�U�\�a�c�a�U�U�S�H�C�D�E�H�H�H�H�H�H�/�<�H�N�T�O�H�A�<�1�/�,�#����#�(�/�/�������������������������������������������(�)�-�0�0�(���
�������������������������õìáåìù�������5�N�Z�g���������������s�Z�N�A�>�;�9�0�5āčĚĦĲĳ����ĳĭĦĚčĉā�v�r�r�{ā�#�<�K�[�q�v�t�h�f�U�<�#��������������#�/�;�H�T�Z�d�a�T�N�C�;�.�"�������"�/�����ûл߻����ܻлû���������������ƎƚƧƳ��������ƳƧƚƎƄƅƎƎƎƎƎƎ��"�.�3�2�.�"���	���������	������������׾ԾʾʾƾɾʾӾ׾������)�B�N�[�g�t�t�u�t�q�g�[�N�B�5�)�!��%�)�T�m�������������y�`�T�G�6�/�2�1�4�;�G�T�F�S�c�l�o�_�Q�B�)����������!�-�F�������	�������ּռмӼּ������FF$F1F=FQFfFmFqFoFcFCF=F<F1F0F$FFFF�����ûʻû����������������������������������������������������������������������s���������������������s�p�f�b�^�]�e�s�y������������������������������y�r�x�y�T�a�b�m�p�z����������z�m�b�a�X�T�R�T�T��������������������������������������������'�4�@�N�V�M�J�@�4�(�����������"�/�2�;�<�>�;�4�/�"��	���	�������������������������������������������3�'�'���	��������'�3�6�=�@�C�A�3��������������������ŹŭũŢŧŭŹŻ����ÇÓàìù��������������ùàÓÇÂ�{ÁÇ���x�l�_�S�F�B�F�S�V�_�l�x��������������D{D�D�D�D�D�D�D�D�D�D�D�D�D�D{DuDrDqDuD{���
�����
�������������������������� e T k H : : G C E = 8 X  d / [ 6 � @ < \ 2 2 / " > 9 S Y o / ) n F J l o $ F r K Z - d 9 : @ f  7  �    =  �  q  �  H  H  �  �  +  B  �  /  �  �  X  �  A  S    �  $  �  �  �  �    �    �  �    e  �  H  y  �  �  (  \  �    �  �  `  b  �  �  E���
�D����o<t�<T��<�t�;�`B=8Q�=��=�hs=\)<�9X=o=q��<�j<�j<�1<�1=�P<���<�/=�\)=e`B=H�9=�\)=8Q�=L��=�P=\)=��=P�`=�E�=�O�=D��=�%=@�=D��=e`B=q��=y�#=�j=��=�7L=�%=�l�=�>I�=�
=>H�9=��#B	��B�TB�IB��Ba�BQeB��B��By�B��B��B!��B	�B�B O�B��B)�B@8B�'B&�vBߴB�yB��B6cBztA�n|B_�B/B� B�VB�B��B��B$��B�yB�uB��B)�B,�GB �Bn�B�JB�2BrVBo�A�_B L&B-�B��A�>�B	��B�B�B��B�	B=�B��BA<B@�B�7B��B")�B$�B��B R8B>�B?B�B��B&��B�B��B��BƊB�A�w1BB{B*{B��BOB��B�:B_mB%<:B�lBP�BC�B�7B,�MB �YB8cB�4B^B�MB�A�z�B ?_B,��B�TA�s�A�zAe��B�S?�M�A�m�A��~@��2Ai�;A�J�A���A ��@��HA�mrA�>?O� A>�"A�"AA���A���@�+A�<XA��A�kA�8/A�B�A�n�@�<UB�qA]LrAS%�A�^Ai�/@s��AHC��}@�b�A�%#AE�A�JA�a�A�u�@�LA�T�A��C?�,VA���A̎b@��SC�ǏA�"A�n�Aeb Bǘ?��A�{�AԃA@�eAj��A�mA�|A�@��A�<A��?P�A? �A���A�yvA­�@�oA��A�rA���Aރ�A��A�wr@��BgA]�AS #A�v�Ai�,@w��ARpC��
@�L�A��AD��A��A�x�A���@�<A��A�d�?��%A�e A�5n@���C��A蜆                        (   e   =            .   
                     1   #      0                     7   #                        -      
             )      X   	                  !      (   )   3            1                        #   %      /                     %   %                        '                           
                              !            /                                 %                        %                                                   
N�ѾNy�dNG�ND�O�+�O�s;N0J	Om[VN�TKOܞ`OF�O�HO��P��N.}N�Z�NG�6NDjN�^3N7�N��TN��,O��O0�P��Nm�UO��N��N�*�NU�:O��O[T{O�*"N�XmONON'"NR�O��O+�0N�'�N�!8Nެ�N�+�Nd�%O0M&O�O��dN�
mON�N"�~    �  �  8  �    \  �  
�  F  �  <  �  �  E  �  �  +  �  �  0    B  �  (  :  r  �  �  �  k  Z  �  �  Y  �  �  �  �  9  �  �  <  ~  �  Q  �  �  �  ��`B�ě��ě��o��o;o;�o<���=�o=o<u<D��<�t�<�1<�o<�C�<�C�<�C�<ě�<�1<�9X=8Q�='�<���=+=\)<��<��<��=+=��=u=\)=��=��=0 �=49X=@�=8Q�=L��=�t�=�o=m�h=m�h=��=��=Ƨ�=���=�l�=�x�qljhit{�����ytqqqqqq����������������������������������������opt�������toooooooo�������� ����������)5BJGOQ\]NB5)WU[^hqoih[WWWWWWWWWW�
#'/;CIJH>/#
��������������������������������������>9768=BFO[bgiifcXOB>����������������������������������������,*-<Hn��������zaUH<,���������������������")2*)��������������������������COT[fhktttrh[OCCCCCC��������������������;:<IUWZXUID<;;;;;;;;����
#')&#
�������!##'/<HIQNHD<8/.%#!!DDCDHHUagnnpnmeaUHDD�������������������)BNTWWNB5���"#/9;<<;/""��������������������"$&)5=>@>85)""""""����������>ABO[_fe[ONB>>>>>>>>)56:8775.)�������
���������)6?HHB;)��� 

#/0870.#
��������������������\\Zadknrnnma\\\\\\\\MHGNV[dghhge[NMMMMMM�������������������������������������{�����������{{{{{{{{��������ZSQWbhntx����|tph[[Z������������������)+)&,664)�����),`^`amz������zmhaa``gbabdhmz���������zmg�����������������������������

�����",-$"�U�a�n�zÇÍÇÂ�z�n�a�[�U�K�U�U�U�U�U�U�G�R�T�\�]�U�T�S�G�;�;�C�G�G�G�G�G�G�G�G��������������������������������������L�Y�a�e�g�g�e�Y�S�L�F�F�L�L�L�L�L�L�L�L�����������������������s�o�s�s�o�u���������)�4�@�J�M�B�)����������������������������������������������������������m�y�����������y�m�`�T�P�G�C�B�F�O�T�`�m�6�B�O�S�[�\�[�Z�O�M�B�6�,�)�&�'�)�/�6�6��)�B�N�[�g�k�[�N�6����������������y�������������ĽȽȽĽ����������y�q�s�y���������� ���������������"�*�/�0�2�/�)�"��	�����������	�
���a�z�����������������{�y�|�z�s�V�O�M�S�a��	�����������������������Z�a�f�j�i�f�b�Z�M�L�H�M�O�V�Z�Z�Z�Z�Z�Z�����	�	��	�����������������������������H�L�U�\�a�c�a�U�U�S�H�C�D�E�H�H�H�H�H�H�/�<�H�L�H�H�<�<�/�$�#�!�#�$�/�/�/�/�/�/�������������������������������������������(�)�-�0�0�(���
���������������������������������������������Z�g�s�����������������s�h�g�Z�V�Q�W�Z�ZāčĚĞĦįĳĺĽĳĦĝĚčā�y�v�u�~ā�<�I�Q�\�c�f�d�Z�W�I�<�#�
���������	�#�<�/�:�;�F�;�3�/�"� ��"�"�/�/�/�/�/�/�/�/�����ûлٻܻ��ݻܻлû���������������ƎƚƧƳ��������ƳƧƚƎƄƅƎƎƎƎƎƎ��"�.�3�2�.�"���	���������	������׾�����׾ʾȾʾʾվ׾׾׾׾׾׾׾��7�B�N�[�g�i�n�i�g�[�N�B�5�2�)�&�%�)�5�7�T�`�m�z���������y�m�`�T�I�G�D�C�E�G�M�T�F�S�c�l�o�_�Q�B�)����������!�-�F������	���������ؼּҼռּ����FF$F1F=FJFPFeFkFoFcFVFJF=F2F1F%FFFF�����ûʻû����������������������������������������������������������������������s���������������������|�s�g�b�b�f�k�s�y������������������������������y�r�x�y�a�m�m�z�}���z�m�e�a�Z�T�a�a�a�a�a�a�a�a�������������������������������������������'�4�@�H�M�Q�M�C�@�4�.�'��������"�/�2�;�<�>�;�4�/�"��	���	�������������������������������������������3�'�'���	��������'�3�6�=�@�C�A�3��������������������ŹŭũŢŧŭŹŻ����ÇÓàìù��������������ùàÓÇÂ�{ÁÇ���x�l�_�S�F�B�F�S�V�_�l�x��������������D{D�D�D�D�D�D�D�D�D�D�D�D�D�D{DvDsDsDwD{���
�����
�������������������������� L R k J : 6 G 4 ! 1 . X  j ? [ 6 � ; < \   %  3 2 S Y P >  n B H l o  F k @ M - d 9 : @ f  7  �  �  =  o  q  T  H  �    �  �  B    \  Q  �  X  �  �  S      ,  q  �  q  H    �  �  P  �      �  H  y  V  �  �        �  �  `  b  �  �  E  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  �  �  �  �  �  �      �  �  �  �  �  �  �  �  {  B  �  j  �  �  �  �  �  �  �  �  �  �  �  �    y  r  _  H  0      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �     /  :  ?  3  $      �  �  �  �  �  �  �  �  f  H  )  	  �  �  �  �  �  �  �  {  l  [  X  `  `  R  ?  *    �  �  �        �  �  �  �  �  �  �  �  s  e  T  @  *  �  �  �  �  \  \  ]  ]  ^  _  _  ^  [  Y  V  S  Q  S  _  j  v  �  �  �  x  �  �  �  �  �  �  �  �  �  d  /  �  �  C  �  j  	  �  �  &  �  	p  	�  
  
C  
B  
.  
S  
�  
�  
�  
�  
z  
  	�  �  �  �  	    "  ,  5  5  9  C  D  :  #  �  �  �  M     �    a  �  �  _  �  �  �  �  �  �  �  �  {  \  8    �  �  X    �  �  B  <  .    
  �  �  �  �  �  �  �  }  b  @    �  �  �  �    f  w  �  �  �  �  �  �  �  w  g  N  +    �  �  '  �  -  �  P  �  �  �  �  �  x  O    �  q  �  -  �  �  �  W  �  F   �      0  :  C  ?  8  (    �  �  �  �  �  �  l  H  "  �  �  �  �  �  �  �  z  q  ^  H  1    �  �  �  o  @     �   �   u  �  �  �  �  �  �  �  �  �  �  �  �  �    t  d  R  @  .    +  )  '  %  #  "  %  '  *  -  )          �  �  �  �  �  0  Y    �  �  �  �  �  �  �  �  m  F    �  �  �  R    �  �  �  �  �  �  �  �  �  �  �  �  �  z  o  d  Z  M  A  4  (  0  *  $          �  �  �  �  �  �    l  Z  P  I  B  :    p  �  �  �  �  �  �  �     �  �  �  �  b    �  �  �  9  n  �  �  �  �         /  ;  A  A  2    �  �    �  ;  �  �  �  �  �  �  �  �  �  o  V  8    �  �  a    �  "  M  �          (  $      �  �  �  �  n  I    �  �  >  �  b  �  �  �  �  �  �  �    .  9  6  '    �  �  �  i  +  �  �  O  `  j  o  r  p  n  g  \  M  ;  $    �  �  �  U    �  R  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  h  c  �  �    �  �  �  �  �  �  �  �  �  �  �  �  p  `  P  =  )       �  z  �  �  �  �  w  i  V  A  +    �  �  �  �  |  Z  5     �  9  F  S  \  d  j  i  d  V  @  #  �  �  �  �  X  :  B  n  �  �  �    *  <  F  N  T  X  Z  X  P  >    �  �  7  �  r  �  �  �  y  �  k  J  S  :    �  �  f  *  �  �  �  8  �  j  �  �  �  �  �  �  �  �  �  �  �  �  t  Q  +    �  �  �  c  *  :  W  P  E  7  &       �  �  �  x  I    �  �  .  �  T  �  �  �  }  t  k  a  V  J  ?  4  )        �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  d  M  3        �   �   �   �   y   ]  �  �  �  �  �  �  �  �  �  �  �  h  N  2    �  �  �  [  
  �  w  c  R  D  4  "    �  �  �  �  �  k  U  F    �  �   �      %  6    �  �  �  �  c  ?  !    N    �  .  �  >  �  y  �  �  �  �  �  �  �  �  �  �  �  �  �  P    �  )  �  �  9  9  ;  A  I  w  �  y  _  8  �  �  p  (  �  �  \  "  �  �  <  3  )    
  �  �  �  �  l  I  %    �  �  �  �  h  a  Y  ~  |  z  x  t  h  ]  Q  C  3  #      �  �  �  �  �  �  �  �  R  +  $  6  �  �       �  �  |    �  ^    �  W  �    Q    �  �  �  �  �  �  y  W  /     �  f    �  F  �  n  �  �  z  \  K  2    �  �  �  �  {  N    �  A  �  �  3  [  #  �  �  �  �  �  �  t  b  P  >  ,      �  �  �  �    v  �  �  �  �  �  �  `  &  �  �  E  �  n  �  :  `    	i  �  �    �  �  �  |  X  3    �  �  �  p  L  )    �  �  �  �  �  �
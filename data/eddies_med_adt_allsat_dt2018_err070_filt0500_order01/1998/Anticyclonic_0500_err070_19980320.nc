CDF       
      obs    3   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?ə�����      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M� '   max       P���      �  x   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �8Q�   max       =�"�      �  D   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?���
=q   max       @E���Q�     �      effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�=p��
    max       @vx��
=p     �  (   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @R@           h  0    effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�o�          �  0h   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���   max       >["�      �  14   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�|�   max       B,��      �  2    latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��Z   max       B,�9      �  2�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?K�   max       C�v�      �  3�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?O�F   max       C�Yy      �  4d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  50   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min          
   max          A      �  5�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min          
   max          5      �  6�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M� '   max       P��E      �  7�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��Ϫ͞�   max       ?�s�g��      �  8`   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �8Q�   max       >�      �  9,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?���
=q   max       @E���Q�     �  9�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�33333    max       @vx��
=p     �  A�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @R@           h  I�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�          �  JP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D�   max         D�      �  K   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��s�g�   max       ?�rGE8�5     �  K�            -   .             q            J            �               $   �      	   N      
      3                        #   �         !   %   )         	         K   NX�N�ʯN/��Puy�P>� O\nJN	yvOĴ,P"�N�E�O9�
O��P���Nx9�N?��O)TP���NT�OɯEN3O�Ob.�O��*Pq�Ne��N͓P<�nOaO9�N���O��2O��sN9�O�PNH`ZN�&O;�M� 'O]�TP�N!��N�{vO�/�O]�O�AzO���N��qN��vO$b�O%@�O:��OQhN�8Q����`B�o$�  :�o;�o;ě�<o<o<t�<#�
<49X<49X<T��<T��<u<�o<�o<�o<�C�<�9X<���<���<���<�`B=o=o=o=+=C�=��=�w=@�=D��=T��=Y�=Y�=]/=e`B=u=}�=}�=�%=�%=�7L=���=���=� �=Ƨ�=�"�rst�������utrrrrrrrr{vvy~�������������{{�����������������������)N\bdlusgN����#1<Uanv}}����n</$��������������������%)6BDKB6.)%%%%%%%%%%SSVaenz���������zaWS���������

������BABEKOX[_a[[OBBBBBBB403<BHUachikkeaUH?<4���������� �����������������-(����������������������������������������������#*,0:<IKHIKB<0($��)6L[t�vvohaOB
aamrz{��zmgaaaaaaaaa��
"*//:??P2/#����������������������c_`aeht����������tnc		
#<HQZ^ZUK<<#	 ��)5VkvzveU='+$(/<CHJHF</++++++++khin{}����������{qnk������)5BLRNB)��������)*2+)��������������������������������������������)'(,:Fan�����nZNH<5)������

�������7<<<HUU]UH<<77777777#0:<HA<700#���������������������������������������������������
����
ZVUY[^htv������th^[Z��������������������	

#$,#
								825<?HQUSJH<88888888��������������������������!%$�������������������������������)5<5)���ywyz~�����������yyyykebbfmuz}}zzomkkkkkk��!,/0) �����ropt�����������~|zxr��������
 �������������
�����ÓàääàÓÓÇÀ�ÇÊÓÓÓÓÓÓÓÓ�g�t£�t�n�g�d�e�g�g��)�1�-�.�)������������������������������ŠŇ�{�b�U�M�L�UŠťŭ��������/�9�C�C�;�/��	��������������������������������������ŭŠŔōŔŠŨŭŹ��ÓÙÕÖÓÇÀÀÇÏÓÓÓÓÓÓÓÓÓÓE�E�E�FFFF"FFF	E�E�E�E�E�E�E�E�E�E���������������������������������������������������������������������������������������������������s�o�p�x�����Ƴ��������������ƳƧƚƎƁ�zƁƇƎƧƮƳ���������������������Z�A�5�����5�Z�����������������������������������������Ź����������������������������������������������������������r�g�_�f�r��f�����Ƽϼϼʼ����w��Y�4�������M�f����������������������������������������y�����������������y�m�G�"����G�T�m�y�����������޻߻�����������(�A�M�Z�]�g�j�f�Z�A�(������������������������������������������B�[�h�tāā�t�h�[�O�B�)��������������zÇÓÖ×ÓÎÇ��z�u�y�z�z�z�z�z�z�z�z�нݽ����������������ݽнĽ��ĽȽпG�y�����������������u�m�`�T�B�7�5�A�A�G�f�s�x�t�s�r�p�m�g�f�Z�M�J�G�K�M�O�Z�]�f�T�`�m�y�|�����������y�m�`�G�@�;�8�@�M�T�.�2�1�*�&�"���	�����������	���"�.����(�A�N�\�[�A�5�(�������߿ҿ�������������ȾǾ�����������p�k�q��������������������������������������������������ݼ��㽫��������������������������������������àâìùúùôìàÓÇÆÄÇÓÙàààà��"�.�;�G�Q�X�T�G�@�;�.�"�����	��āā�t�o�m�tāĆćāāāāāāāāāāā�!�-�:�S�Z�_�g�a�_�S�?�:�-�!������!�M�Y�f�g�_�@�0�����ۻһ̻ǻʻм�'�@�M�f�s�u�������s�o�f�b�e�f�f�f�f�f�f�f�fD�D�EEEEEED�D�D�D�D�D�D�D�D�D�D�D�Ħĳ������������������ĳĥĚĘęėĚĞĦ���������
������
�������������������H�U�]�a�f�n�xÅÊÇ�z�a�H�8�/��)�/�B�H¿���������
���������������½º¿��(�5�A�N�S�X�N�A�5�(���������ŠŭŹ������������ŹŭŬŠŞŠŠŠŠŠŠ�b�i�i�V�N�0����������$�0�=�I�V�]�b�����	��������������������������������D{D�D�D�D�D�D�D�D�D�D�D�D�D{DwDnDlDoDqD{��������������ɺ������ɺԺֺ�� O 0 E E . V n  ) 7 & g 8 U * 0 Y 6 I - C . = 7 u : n N t E 2 b 3 N 7 & Z 1 S N .  = D P T R � | 6 e    q    Y  D  ?     ]  �  �  �  �  b  �  �  J  [  �  o  �  M    J    o    R  �  q  J  e  <  q  !  w  �    1  �  �  S  �    �  m  �    �  �  �  �  ����ě�<e`B=,1=<j<�9X<#�
=��=�<e`B<��<�C�=�1<�j<��
<���>["�<���=#�
<�9X<�`B=m�h> Ĝ=+=\)=��=,1='�=�P=�1=u=,1=H�9=P�`=m�h=��=q��=�9X>P�`=y�#=���=��=���=��=�j=���=�1=\=���>/�>I�B
TCB
��B��B��BX�B��B�CBk%BF]B��B�AB�B�BPB �B%ٟB�[A���B�FB"3B��Bg�B5�B�B)�B��B��B �B"HB�uB�B<CB%bB,��B!�BKB��B��B�OB$�!BB-JB�B��Bn�B
�A�|�BO4B
c8B�zB�B
��B
��BBBxB>�B�TB�wB�LB@�B��B�B@�BB�B?�B�3B%ʄBG A���B��B"�B>�B�*B@�B�&B(�UB�DB��B��B"EIB��B<�B�B%@dB,�9B""�B@�B]0B�&B�jB$�EB>�BA�BA�B:�B�B
�BA��ZB�B
�2B��B�A�QA���A�vA���A��^A��kA�%�C�v�A��/AIa�AH�|B��A��pA��|?K�@�Z%@���A�$�Ag�U@�;�A9Z�AҜMA�5�A�y}A,Al/�A@dAh�OA]�A�Y�AI��A2V�A��A$!�A�SFA`��A�|@}��@�YAB�C�O�A�ZA��AŮuA�m�A�8�A��DB
�A�wC��h@L�A�t A�W�A�~�A���A���A�z�AɀC�YyA�qZAIc>AI�B$�A�vA�uw?O�F@��@�0�A��Ag��@�JA7JAҀA�~dA�}rA)oAl�%A>�bAgBA[�A��fAK-�A2�MA��A$�A�"+A`��A܋�@|(�@�xAB�C�W�A��A�W�A�o#A�|A���A�I�B
?JA��/C��1@La            -   /         !   q            J            �               %   �      
   O      
      4                        $   �         !   &   )         	         K               5   -         !   '            ;            A      '            3         ,            )                     
      /                  !                              !   '         !               5                  %            +                     )                     
                        !                  NX�N�ʯN/��O�9P+�O0N	yvOĴ,O��N�E�O	��O��P��ENQa�N!�QN�)�O`JNT�O�=�N3O�Ob.�O(�GP9��N)u�N͓O�p�OaN���N���O��2O}|�N9�N��gNH`ZN�&O;�M� 'O�YOo�tN!��N�{vO�/�OS�O��!O���N��qN��vO$b�O%@�O7e�O77  Y  O  �  �  R  @  (  �    �  �  �  �  �  O    �    �  J  B    8  �    	  �  �      �  Q  �  �  �  �  �  �  �  /  M  Q  	  �  J    �  P  �  �  �8Q����`B<�C�<t�;�`B;�o;ě�=P�`<o<T��<#�
<�t�<D��<e`B<�t�>�<�o<�C�<�o<�C�=+=<j<�/<���=T��=o=C�=o=+=t�=��='�=@�=D��=T��=Y�=u=�S�=e`B=u=}�=�%=��=�%=�7L=���=���=� �=ȴ9=�;drst�������utrrrrrrrr{vvy~�������������{{�������������������� )BNSVWde`[B5!! (<HUakut{zmH</+#!��������������������%)6BDKB6.)%%%%%%%%%%SSVaenz���������zaWS����������	������BABEKOX[_a[[OBBBBBBB8<=FHU_affhgaUHF<<88���������� ����������������&!����������������������������������������������� #%0<=DE=<20.*#    )6HNPNGB61)aamrz{��zmgaaaaaaaaa��
#/9>>NHF</#����������������������c_`aeht����������tnc #/<GHPUOH?</#!
	)5Kanso^TJ5
(-/<@HIHC<1/((((((((khin{}����������{qnk������);><.)��������)*2+)��������������������������������������������)'(,:Fan�����nZNH<5)������

�������7<<<HUU]UH<<77777777"#+0<><;50+#���������������������������������������������������
����
ZWY\dhnt|�����|th_[Z��������������������	

#$,#
								825<?HQUSJH<88888888�������������������������� $#��������������������������������)5<5)���ywyz~�����������yyyykebbfmuz}}zzomkkkkkk��!,/0) �����ropt�����������~|zxr��������
 �������������	�����ÓàääàÓÓÇÀ�ÇÊÓÓÓÓÓÓÓÓ�g�t£�t�n�g�d�e�g�g��)�1�-�.�)���������������{ŠŭŹ������������ŹŠ�{�n�b�_�\�_�n�{������/�6�9�8�/��	��������������������������������������ŹůŭŦŭŮŹ��������ÓÙÕÖÓÇÀÀÇÏÓÓÓÓÓÓÓÓÓÓE�E�E�FFFF"FFF	E�E�E�E�E�E�E�E�E�E���������������	��������������������ž��������������������������������������������������������������u�u�}���������Ƴ��������������ƳƧƚƎƁ�zƁƇƎƧƮƳ�s���������������s�R�A�5�"����#�5�Z�s���������������������������������������ҹ������������������������������������������������r�p�f�e�f�r�{�����r������������r�f�Y�M�@�=�=�@�D�P�Y�f�r����������������������������������������y�������������y�m�G�@�;�/����G�T�m�y�����������޻߻�����������(�A�M�Z�]�g�j�f�Z�A�(�����������������������������������������)�O�[�h�p�|�|�t�h�[�O�B�)����������ÇÓÔÕÓÉÇÄ�z�v�z�|ÇÇÇÇÇÇÇÇ�нݽ����������������ݽнĽ��ĽȽп`�m�����������������m�`�T�G�D�B�D�N�T�`�f�s�x�t�s�r�p�m�g�f�Z�M�J�G�K�M�O�Z�]�f�T�Y�`�m�t�y���y�q�m�`�T�P�G�>�E�G�S�T�T�.�2�1�*�&�"���	�����������	���"�.����(�A�N�\�[�A�5�(�������߿ҿ�������������Ǿž�����������v�s�n�s�v������������������������������������
��������߼���ｫ��������������������������������������àâìùúùôìàÓÇÆÄÇÓÙàààà��"�.�;�G�Q�X�T�G�@�;�.�"�����	��āā�t�o�m�tāĆćāāāāāāāāāāā�-�:�F�S�_�b�_�\�S�H�F�:�/�-�!���!�(�-�����'�2�1�*�����������ۻۻܻ���f�s�u�������s�o�f�b�e�f�f�f�f�f�f�f�fD�D�EEEEEED�D�D�D�D�D�D�D�D�D�D�D�Ħĳ������������������ĳĥĚĘęėĚĞĦ���������
������
�������������������H�U�[�a�m�wÃÉÇ�z�a�H�=�/�#�%�+�/�D�H¿���������
���������������½º¿��(�5�A�N�S�X�N�A�5�(���������ŠŭŹ������������ŹŭŬŠŞŠŠŠŠŠŠ�b�i�i�V�N�0����������$�0�=�I�V�]�b�����	��������������������������������D{D�D�D�D�D�D�D�D�D�D�D�D�D{DwDnDlDoDqD{���������������ֺɺºɺ˺պ�� O 0 E @ + < n   7 $ g 7 S ) * : 6 H - C ) 6 : u * n + t E / b + N 7 & Z +  N .  8 @ P T R � | 6 T    q    Y  /  �     ]  �  +  �  -  b  L  �  2  �  �  o  �  M    k  /  Q    �  �  �  J  e  �  q  �  w  �    1  .  �  S  �    �  Y  �    �  �  �  �  �  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  Y  W  T  Q  I  A  7  *        �  �  �  �  �  �  �  �  �  O  G  ?  8  &    �  �  �  �  �  �  a  F  @  9  7  N  e  |  �  d  z  �  �  )  B  �  	/  	�  
K  
�  
�  
�  !  O  }  �  �      :  P  d  k  Z  6    �  �  {  V  '  �  �  '  �  �  ^  ,  	  3  G  Q  R  P  B  +    �  �  �  �  Y    �  =  �  �  *    '  &  (  0  <  ;  1  !    �  �  �  �  X  '  �  �  Z  �  (  !          '      �  �  �  �  �  �  �  �  �  �  �  �  �  f  ;    �  �  �  �  `  )  �  �  �  �  �  |  K    �  	  	�  
�    z  �  �      �  �  �  ;  
�  
C  	�  �  X  e    �  �  x  o  c  W  K  =  /       �  �  �  �  �  h  J  -    �  �  �  �  �  �  �  �  �  �  l  C    �  Z    �  x    �  �  �  �  �  �  �  �  �  ~  p  b  T  F  7  &       �  	   }  �  �  �  �  �  �  �  �  �  j  ;    �  �  f    �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  A  N  N  O  O  N  L  K  I  H  =  1  #    �  �  �  �  �  f  D          	            �  �  �  �  |  N    �  �  N  X  �  �  �  ?  �  I  �    h  �  �  �  �    �  t  p  �  
    
    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  �  �  �  �  �  �  �  �  �  v  T  &  �  �  m  #  �  {  �  _  J  A  9  0  )  #             �  �  �  �  �  �  �  �  �  B  9  /  %        �  �  �  �  �  �  �  �  d  A  )     �  ^  �  �  �  �         �  �  |  A    �  �  E  �  -  R  ^  �    /  7  1    �  �  k  8  '  �  �  U  
�  	e    �  �   �  �  �  �  �  �  �  �  �  �  �  u  `  H  0    �  �  �  �  �    �  �  �  �  �  �  �  �  �  n  J  %  �  �  �  I   �   �   �    D  �  �  �  �  	  �  �  �  v  *  �  q  �  ]  �  �  �  �  �  �  u  [  D  9  *    �  �  �  �  �  �  l  :  �  �  v  ,  �  �  �  �  �  �  �  �  �  �  q  [  B  &    �  �  �  �  w         �  �  �  �  �  �  �  �  �  t  e  W  H  2       �    �  �  �  �  ~  V  '  �  �  �  B  �  �  Z  �  \  �  "  �  �  �  �  �  �  �  �  m  N  0    �  �  �  ~  ;  �  �    r  Q  ;  %    �  �  �  �  �  �  z  e  P  ;  &    �  �  �  �  {  �  �  �  �  �  �  �  |  s  g  X  G  3      �  �  �  p  �  �  �  �  �  �  �  �  �  �  �  ~  y  u  p  e  Y  L  @  4  �  �  v  _  H  4  "    	  �  �  �  �  �  �  �  �  g  A    �  �  �  p  V  <       �  �  �  k  :    �  �  Y    �  b  �  �  �  �  �  �  �  �  �  �  �  |  s  i  [  L  I  z  �  �  f  �  �  �  �  �  �  �  �  �  m  >    �  k  �    /  ;  2  7  =  �    f  �  �  �  �  �  .  �  �       �  m  	�  O  �  /  '      
  �  �  �  �  �  {  `  C  %     �   �   �   �   j  M  %  �  �  �  v  A    �  �  a  #  �  �  h  (  �  ~  �  s  Q  L  7      �  �  �  l  9  �  �  \     �  D  �  A  �  �  	  	  	  �  �  �  |  L    �  �  -  �  b  �  r  �  W  �  _  z  �  �  �  �  }  k  U  ;    �  s    �    ~  �  x  �  �  J  =  "     �  �  m  2  �  �  c    �  �  �  >  �  m  =  8       �  �  �  �  x  X  8    �  �  �  �  s  N  +      .  �  �  �  �  �  �  �  �  �  q  a  O  :    �  �  O  �  �  $  P  "    �  �  �  �  �  �  �  j  ?    �      �  	  �    �  �  �  r  W  T  \  K  :  *    �  �  �  �  i  A    �  �  �  �  �  �  [    �  Y  �  �    �    [  t  
j  	>  �  �  �  �    �  �  �  �  c  @    �  �  c    �  ?  �  b  �  u  �
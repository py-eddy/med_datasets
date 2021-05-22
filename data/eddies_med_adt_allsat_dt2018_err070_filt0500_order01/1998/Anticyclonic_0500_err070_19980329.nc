CDF       
      obs    8   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�n��O�<      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       P�L�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       =Ƨ�      �  l   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?���Q�   max       @F���R     �   L   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�33332    max       @vu�����     �  )   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @N�           p  1�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��@          �  2<   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���   max       >L��      �  3   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�j!   max       B3��      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�jC   max       B3��      �  4�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?,�   max       C��'      �  5�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?-?_   max       C��,      �  6�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;      �  8\   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          5      �  9<   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       PtB      �  :   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�ح��U�   max       ?�L/�{J$      �  :�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��P   max       =���      �  ;�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?���Q�   max       @F�\(�     �  <�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�33332    max       @vu�����     �  E|   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @N�           p  N<   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�           �  N�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D�   max         D�      �  O�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?}��,<�   max       ?�L/�{J$     �  Pl                                 #               9   q   U   l         	               ?            :            	         5      ,      "   	      	      >   >         �               O5��NG{)N��?N�kXOw�N2*mM���N�_-N�\&Ok�Pb�O_/N�ȌO]|�N]�]PI�PS�P9��P!veO JZO��N#�N8{Ni��N5T�NK	�P�L�N�R�N�T�OXy}PS�[NT��N���O��O�N�#/N[!�Ohs�O(�O��gN�� O�J�N���OP�FO-GO���O��rO��ZOoO~��PN�,MN���O�DN
��N	ȼ�����T���D�����
�D����o��o��o;o;D��;�o;ě�;ě�;�`B<o<o<49X<D��<D��<e`B<e`B<u<�o<�o<�C�<���<�1<�1<�1<�9X<�9X<�9X<�j<�j<ě�<�`B<�h<�h<�=o=+=\)='�='�=49X=@�=@�=L��=aG�=e`B=q��=}�=�7L=�C�=Ƨ���������������������ppt}�������tpppppppp��������������������rot��������{xtrrrrrr���� # ����������������������llnxz����zsnllllllll�������������������������������������������������������������������)8)���������������������������;767@BGORX[^_^[OFB;;dieht������������thd..06@BHJGCB6........����
/=EHPUWR></#��$!!#)6[������thOB60$#/Han������aH</#��������

�����-.*./<HU[a\XUJH<5/--)5BLNVXQNEB5)(��������������������36<HHUUUUUH<33333333�������������������������������������������������������������������������������������������������/'(/<HSU\[UH<///////����������������������������1=IE)�����������������������|||������������||||��������������������������� 

����������������������������������������UTSQH</##-/<HJUU����������������������
<HOU\]XH=</!
���������

���������������������������������

��������*57;:85)�������������������������

�����~|�����������������������),.*����kkmwz�����������zmkkKS[gt��������tvni[RK����������������������������������������LOQW[hjoqphh[UQOLLLL),6BCB@?962)��������������������a_amrspmjaaaaaaaaaaa�������
���������������������������L�Y�^�e�j�e�`�Y�P�L�C�E�L�L�L�L�L�L�L�L�T�`�a�c�l�m�q�m�g�a�T�L�H�G�H�I�T�T�T�TÓàëêàÝÓÇ�|�z�x�zÇÍÓÓÓÓÓÓ�F�S�_�x���������������x�l�_�S�F�:�5�<�F��)�/�,�)�'��������������E�E�FFFFFFE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eٹ��������������ܹϹιϹֹܹ�������#�(�4�4�*�(�����������������	�"�;�N�N�@�;�-���������������������������������������������������������Ҿ���������������������������}�|��������Z�f�s�u�t�{���������������s�f�K�?�H�O�Z�����ʼѼּؼּʼ������������������������`�l�������������y�l�`�T�G�;�"����5�`����'�Y�j�|�u�t�f�@�4����ٻԻλջܻ��5�N�f�o�j�h�c�K�6�5�(��
�����������5���������������������������������������	����"� ��	��������������������Ź��������������ŹŭŠŖŠŢŞşŠŭŴŹ��������������¿²±²³¿�������������˾������������������������������������������(�0�5�7�5�(�����������������������޻�����������ÓÖÞÛÓÇÁ�z�w�zÇÑÓÓÓÓÓÓÓÓ�Z�s�����������������g�N�(���ڿѿ���Z�a�b�h�m�k�a�U�L�H�C�C�H�U�^�a�a�a�a�a�a����!�!������� ��������������ּ�������ּϼʼ����������������;�G�m�y�������������y�m�T�;�.�)�.�1�1�;ƧƳ������������ƳƧƞƜƧƧƧƧƧƧƧƧ�����	��	�������׾־վ׾ؾ�����"�4�7�8�.�(��	���۾׾Ӿվ׾����������׾ھ���׾ξʾ������������������������
����������ݹܹչܹܹ���{ŇŔŔŔŐŇ�{�n�e�n�n�{�{�{�{�{�{�{�{EEED�D�D�D�D�D�D�EEEE*E/E<E8E7E*E������	�����������ֺں��������������!������������������������޾��������������������������������������������������������������y�`�I�K�S�`�l�y���a�n�q�s�zÂÇÈÇ�z�n�e�a�X�W�_�a�a�a�a��������������������¿¹­³¿����������.�;�G�T�`�i�k�`�T�N�G�;�.�"�!��"�*�.�.DoD{D�D�D�D�D�D�D�D�D�D�D�D�D�DnDZDWDbDočĚĦĿ������������ĳĦčā�u�u�v�|āč�������
��(�,�)�#��
�������������������T�a�c�m�q�z�������z�m�c�a�T�T�N�L�R�T�T�"�/�4�7�7�*��	���������������������"�M�Y�f�r���|�f�S�4�'������޻ݼ��4�M�zÇÓßàèæàÓÇ�z�z�v�w�z�z�z�z�z�z�:�B�F�S�Z�V�S�F�:�-�!� ��!�-�0�:�:�:�:�������������º����������������~�����������ʼ˼ּټ޼ּʼɼ���������������������Ź��������ŹŭũŭŹŹŹŹŹŹŹŹŹŹŹ \ . _ E l L ` ' 8 = H U = { K - . I 7 # A r X 9 < ? V 6 Z > 9 ] B - ` O Z 9 / Y X  Y J T +  2 J ] Q / 8 A r f  �  `  �  �  2  a  G  �    >  �  o  	  e  |  �  y  j  �  U  M  n  �  }  E  q  :  �  �  �  �  �  �  I  i  "  e  �  :  �    d  �  �  %  �    �  A  :  �  �  �  9  `  (�����t����
<ě�<�j;o<49X<#�
<o=��<ě�<�C�<D��<D��=��=���=\=�<�9X<���<�9X<�C�<�9X<�j<�j=��T<�`B<�`B=\)=���<���=+=�P=o<��=\)=��T=D��=��=�P=�+=0 �=ix�=L��>/�=�/=�;d=�C�=��P>L��=�t�=��w=��=��=��B!�B�B�'B
[�B�|B��B50Bt�B oB��BBoYB�0BUKB��B�AB�<BőB<�B�B`CB�uB�B�B!��Be�B6�B��B��B!�OB��B�nB3��B!f�B#�iB�EB��B�)B"�B�gB��B,4FB��B"sB%B��B�B�B G�B	�B�oB!�Bq�B5pB� A�j!B=�B��B�6B
J�B��BA�B>mB��B�VB ;�B�B�B�LB�B�B\�B�aBĀB>B�B��B²B8>B��B!ͬBCRBP�B�B
B!��BJ�B~�B3��B!<:B#�}B��B�BH�B"<DB�B B,4�BiBN�BAWB��BK�B;�B ��B
:HB��B!��BJ�B?�B�wA�jCA�$?�޲A�@�A�b@���AչcC��'C�s�?,�A1TA��A��AI�QAC�1@���Ah�/@�JA�; A�c%A��A�$JA�AHRA�� @���A�	�A�6�A���A3(@��Akh�BU�AW ~AZ�|AMqb?/�,A��
C�`�@Q�Aҏ�AJ�AU�Aǵ-A�w�Ad�C�ñA��A�UA� �A���@�~A��@{�p@3"@�ʰA���A��?�N�A�sPAʉ�@��QA�wkC��,C�v�?O�'A0�A��AЃAIjA?6L@��Aif@��_A�YA�~.A���A�~�A���AH��A��%@��yA�y'A�u�A�i�A2��@��,Ak�BE�AX��A[�PAL��?-?_A�i�C�U�@P^`Aҁ�AK&A�*Aƃ�A���AcPC���A��A�~�A�;�A�G�@���Aɇ�@}��@?	@��A�!                                 $               :   r   U   m         	               ?            :            
         5      ,      "   	      
   �   >   ?         �                                                7         #      )   -   -   '                        ;            -                           !      !                           +                                                #         #         !                              5            !                                                                           O��NWSN��?N�kXN�r!N2*mM���N��.N�\&N�ΰO��O_/N�:�O]|�N]�]O���O��O�W�O}��O(O��N��N8{Ni��N5T�NK	�PtBN�R�N�T�O�UO�\�NT��NIW�Ot��N�H�N��-N[!�Ohs�N�Y�N��N�� N��_N���OP�FN��O7�O��>O�QRN�;�Of�O�utN�ҀN���O�DN
��N	ȼ  �  �  ]  "  �    �      7  w  U  5  �  H  �  	�  -  :  �  `  �    �  �  �  �  �  k     W  a  �  �  A  V  �  
f  �  �  }  a  �    f  ~  	�  
�      �  �  �  &  �  ���P��P�T���D��;ě��D����o$�  ��o;D��<��
;�o;�`B;ě�;�`B<�=#�
='�=m�h<T��<e`B<u<u<�o<�o<�C�<���<�1<�1<ě�=#�
<�9X<ě�<���<���<���<�`B<�h=+=H�9=o=H�9=\)='�=,1=�Q�=L��=T��=T��=e`B=���=u=}�=�7L=�C�=Ƨ���������������������sqt�������tssssssss��������������������rot��������{xtrrrrrr����� ����������������������llnxz����zsnllllllll����������������������������������������������������������������������	���������������������������<889BORW[]][UOKB<<<<dieht������������thd..06@BHJGCB6........���
#/9ADD<5/#
�0-**.6Bht|��xrh[OB60-*)+/<HUan~��~naUH9-����������������.///+//<HUZ_ZUSH</..)5BLNVXQNEB5)(��������������������36<HHUUUUUH<33333333�������������������������������������������������������������������������������������������������/'(/<HSU\[UH<///////�������������������������� 28:5)��������������������������������������������������������������������� 

�������������������������������������������UTSQH</##-/<HJUU��������������������!!!#./<BHNRKH<<//#!!�������

���������������������������������

��������*57;:85)������������ �������������

������~�����������������������%*+(����nmyz�����������znnnnMT[gt{�������toj_[TM����������������������������������������LOQW[hjoqphh[UQOLLLL),6BCB@?962)��������������������a_amrspmjaaaaaaaaaaa���
��������������������������������L�Y�]�e�g�e�[�Y�U�L�D�I�L�L�L�L�L�L�L�L�T�`�a�c�l�m�q�m�g�a�T�L�H�G�H�I�T�T�T�TÓàëêàÝÓÇ�|�z�x�zÇÍÓÓÓÓÓÓ�S�_�l�x�|�������������x�l�_�S�F�F�F�O�S��)�/�,�)�'��������������E�E�FFFFFFE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E���������������ܹϹιϹֹܹ�����#��������������������������"�/�8�<�;�>�:�/�"��	���������������������������������������������������Ҿ�������������������������~������������Z�f�s�u�t�{���������������s�f�K�?�H�O�Z�����ʼѼּؼּʼ������������������������`�m�y�����������y�`�T�G�6�.�+�.�:�H�T�`���'�4�M�X�c�c�^�M�@�4������������(�5�A�N�W�[�[�Z�T�A�5�(���	��������������������������������������������������	���� ���	����������������Ź��������������ŹŭŠŖŠŢŞşŠŭŴŹ��������������¿²²²µ¿�������������˾������������������������������������������(�0�5�7�5�(�����������������������޻�����������ÓÖÞÛÓÇÁ�z�w�zÇÑÓÓÓÓÓÓÓÓ�����������������g�N�(��������(�Z���a�b�h�m�k�a�U�L�H�C�C�H�U�^�a�a�a�a�a�a����!�!������� ��������������ʼּۼ�ۼּ̼ʼ��������������������`�m���������������m�`�T�F�?�;�=�@�I�T�`ƧƳ������������ƳƧƞƜƧƧƧƧƧƧƧƧ�����������ھھ�����������"�,�2�3�.�"��	������׾վؾ�����������ʾ׾ܾ׾ʾȾ���������������������������
����������߹ܹعܹ����{ŇŔŔŔŐŇ�{�n�e�n�n�{�{�{�{�{�{�{�{EEED�D�D�D�D�D�D�EEEE*E/E<E8E7E*E���������������ں޺�������������������������������������뾘���������������������������������������������������������������y�r�l�l�q�y����a�n�q�s�zÂÇÈÇ�z�n�e�a�X�W�_�a�a�a�a��������������������¿¹­³¿����������.�;�G�T�`�d�c�`�T�J�G�;�.�%�"��"�-�.�.D{D�D�D�D�D�D�D�D�D�D�D�D�D�D{DoDmDjDoD{čĚĦĿ��������������ĳĦĚčā�w�wāč�������
��&�)�&� ��
�������������������T�a�m�n�z������z�m�f�a�U�T�P�N�T�T�T�T�"�/�3�5�5�/�"��	���������������	���"��'�@�M�U�U�N�C�5�'��������������zÇÓÛàçåàÓÇ�{�z�w�z�z�z�z�z�z�z�:�B�F�S�Z�V�S�F�:�-�!� ��!�-�0�:�:�:�:�������������º����������������~�����������ʼ˼ּټ޼ּʼɼ���������������������Ź��������ŹŭũŭŹŹŹŹŹŹŹŹŹŹŹ Z + _ E b L ` # 8 G > U = { K 2 & -  & A s X 9 < ? H 6 Z 8 0 ] P / \ N Z 9   X - Y J P )  0 N ] 2 % 8 A r f  m  7  �  �  3  a  G  �    �  �  o  �  e  |  L    �  �  "  M  Z  �  }  E  q  i  �  �  B    �  T  �  �  �  e  �  �      �  �  �  �  {  �  ]  #  �  g  �  �  9  `  (  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  �  �  �  �  �  �  �  �  �  �  �  �  |  f  F     �   �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  e  U  D  ]  Y  V  R  O  I  ?  5  *         �  �  �  �  �  �  �  �  "                �  �  �  �  �  �  m  _  P    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  O    �  J  �  E      T  �  	  	t  	�  
J  
�  L    4  "  �  �  7  v  �  �  -  g  �  �  �  �  �  �  x  p  h  `  X  Q  I  A  9  1  )  !      	            	    �  �  �  �  �  �  �  �  �  z  H            	      �  �  �  �  �  y  F    �  �  "   �   b    %  -  5  6  5  5  7  ;  ?  >  7  1  &       �   �   �   �  \  `  _  ]  Z  Z  Z  a  l  v  p  K    �  r    �  �  F   �  U  A  1      �            �  �  �  �  g    �  {  �  3  4  4  /  &      �  �  �  �  �  s  R  3    �  �  �  p  �  �  �  �  �  �  �  �  �  {  k  Y  G  5  %       �   �   �  H  C  ?  :  5  /  )  #               �  �  �  �  �  �  h  �  �    :  Z  r  �  �  k  I    �  �  q    �  �  R    �  	%  	]  	~  	�  	�  	�  	�  	�  	f  	%  �  �  w  J  &  �    @  ?  �    b  �  �    (  -      �  �  o  .  �  K  �  �  E   �  z  	5  	�  
F  
�  
�    0  :  3    
�  
�  
0  	�  �  �  �    �  �  �  �  �  �  �  �  �  �  �  �  {  `  ?    �  �  g  @    `  X  N  @  ,      �  �  �  �  �  �  �  t  n  v  �  �  �  �  �  �  �  �    �  �  �  �  e  D  #     �  �  �  s  N  )    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  ^  A     �   �   I  �  �  �  �  ~  s  g  \  P  G  >  4  "    �  �  �  :  �  �  �  �  �  �  �  �  �  �  z  t  l  c  [  P  C  6  '      �  �  �  �  �  �  �  }  S  %  �  �  n    �  K  �  �  -  ;    �  �  �  �  �  �  �  �  �  �  �    k  V  G  ?  7  "    �  k  g  c  ^  W  O  B  1       �  �  �  �  �  �  j  7  �  �                   �  �  �  �  �  z  a  L  :    �  -  [  �  �  �  &  I  W  M  9    �  �  i    �  m    ]  |   �  a  Z  R  K  D  <  5  /  )  #          
        �  �  r  |  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  G  �  x  �  �  �  �  �  r  b  P  =  (      �  �  �  �  w  ^  B  4  :  @  A  A  =  7  0  )  "    
  �  �  �  y  J     �   �  H  M  R  T  N  H  A  ;  4  )        �  �  �  �  �  t  D  �  �  �  �  �  �  t  c  R  A  /      �  �  �  �  _  /  �  
f  
W  
S  
I  
<  
,  
  	�  	�  	�  	P  �  t  �  s  �  [  �  	  h  k  �  �  �  �  �  �  �  �  �  p  T  ,  �  �  �  t  M  +    /  Z  �  ?  �  �  �  �  �  �  �  �  Z    �  |  �  ,  #    }  }  |  |  z  s  l  e  [  K  :  *    �  �  �  �  �  �  {  ^  s  �  �  �  �  �  !  K  _  _  S  =    �  �  w    Q  !  �  �  �  �  �  �  �  �  �  p  d  ^  [  l  }  u  c  Y  Y  Y    �  �  �  �  �  �  �  �  |  i  W  H  6  "    �  �  �  |  a  c  e  c  a  X  N  @  0      �  �  �  �  e  8     �   �  �     �  �  4  f  y  ~  l  5  �  K  �  �  �  �  s    	  �  	�  	�  	�  	�  	�  	t  	/  �  �  A  �  �  S    �  '  �  �     �  
�  
�  
�  
�  
�  
�  
n  
,  	�  	�  	3  �  D  �    |  �  C  �  �          �  �  �  �  �  o  ?    �  �  k  1  �  �  �  �  �    �  �  �  �  �  �  [  @  /    �  �  v  2  �  �  �  1  �  �  V  �    p    k  >  �  ^  �  H  �  �  �  
�  	H  �  �  �  �  �  �  �  �  �  �  q  Z  @  $    �  �  �  �  e    �  �  t  `  L  6      �  �  �  �  �  �  g  E  #  �  N  �  �  &    �  �  �  �  �  �  �  �  �  �  }  g  P  1    �  �  �  �  �  �  �  �  �  �  g  :    �  �  f  2  �  �  �  Y  !   �  �  ~  r  f  Z  N  D  :  /  %          	  	        
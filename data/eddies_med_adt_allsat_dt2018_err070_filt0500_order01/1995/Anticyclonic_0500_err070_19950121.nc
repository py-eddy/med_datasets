CDF       
      obs    7   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�O�;dZ      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       Mѣ}   max       P��@      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �@�   max       >�      �  d   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?\(�\   max       @E�p��
>     �   @   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�    max       @vP�\)     �  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @O�           p  1p   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�w        max       @�֠          �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���   max       >vȴ      �  2�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B,��      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�z�   max       B,��      �  4t   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?n�   max       C�oJ      �  5P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >���   max       C�o      �  6,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =      �  7�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7      �  8�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       Mѣ}   max       P1�      �  9�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���(��   max       ?�F
�L/�      �  :x   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��w   max       >O�      �  ;T   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?\(�\   max       @E�p��
>     �  <0   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�    max       @vP�\)     �  D�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @O�           p  M`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�w        max       @�L�          �  M�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @H   max         @H      �  N�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?x�PH�   max       ?�C��$�     0  O�   "      �            %   F         (      O               
          r   	            *   '            4      
   .      &            +   
   )                     �   	   	   	         OFRtNH٘P�u�N�^O	\�Nr��P5�O�O&N_zN��`P�1N�P��@O�s�O�]O=Q\O�]�N9%�N��O�3O��N4~�N��O�XN(�dO�$�O��dN��QO�^N�ݞPLR�N2�zO���P ?4OØO�y�N߶bN[�&N�ӭO���N�3�O|�5M��(O1��N4�N]	O��OA6PjNn�LMѣ}N]J�N�0�O��N�W��@��+���u�#�
�#�
���
��o��o;o;��
;�`B<o<t�<#�
<49X<T��<�o<�C�<�9X<ě�<���<�/<�`B<�`B<�h<�h<�h<�h=o=C�=C�=\)=t�=t�=t�=�P=��=#�
=@�=@�=@�=H�9=H�9=H�9=m�h=�o=�7L=�hs=��=��w=Ƨ�=��=���>�TUVTSTW[gty}��|tg`[T��������������������5Ng~�������tg67){{��������������{{{{����������������������������������������������&,/.1)$��������� $&$���!)6BDDB6)��������������������2>O[fjmswxmh[OB:6872#/11///#<Rg����������g[NCC5<��������������������#,5<N[lpmijh[B)���
##/883/*#
���� 
#/HJHHE?</#
��&#)06BFCB6-)&&&&&&&&2546;BO[ad[OB6222222!"$)/<HUamzm]UH</-%!��������
��������������������������LCDHNP[gjttwtrgb[PNL�wutrqrtt������������������������������������������dbdpz������������znd�����������������������)56;:6)��
#
�������������)198:H`sgB)�`gqtx�����~tmg``````}������������������}��)1-+2;=7)���:=66BFN[cgknlga[NB::������������������-++/<HKUWZUSH<7/----��������������������[\hjt�������th[[[[[[��������������������	
#)+(###
				

#(/?B?<:/#�����������������������������������).-,)zxx{�����������{zzzz��������������������''/6=>=BFGO[`bb[O6)'xy|���������������zxPHKTaelhaTPPPPPPPPPP84:<>HIIH<8888888888�

�������������������������������NOSTabmtz|~|znma^TTNHE@HTaelbaTHHHHHHHHH���������
��#�)�6�/�#��
�������������������
����
����������������������������)�6�I�T�M�6�������àÇ�n�o�{Óà���������������������������{�y������������������������������������������������������� �����	�������ݹ����������	��"�;�M�Z�_�/�"��	�����������������H�a�zÇÓÚÙÖËÇ�z�a�U�H�<�6�-�,�0�H�ʼѼּܼ�ټּʼȼ¼üżʼʼʼʼʼʼʼ��B�O�[�c�[�Z�O�L�B�6�)�$�)�)�6�8�B�B�B�B�	�"�%���	��׾����s�j�h�l�z�����ʾ�	EEE#E*EEEEEEEEEEEEEEEE�g���������������Z�N�(��
�������(�L�g�������(�/�8�9�3�$�������ݽ���;�G�T�`�y�������������y�`�T�G�;�3�*�/�;����)�1�/�)�)�"���� ���������������������	�����ݿѿƿ������˿ѿݿ�'�3�?�3�-�0�'������'�'�'�'�'�'�'�'�ݿ�����������ݿۿؿֿؿݿݿݿݿݿ�����������������������������������������DoD�D�D�D�D�D�D�D�D�D�D�DoDaDKDQDQDVDbDo�������������ټ߼����������������������������������������������������������Žy�������������ĽĽ̽ͽĽ����������|�y�y�ܹ������ܹٹѹٹܹܹܹܹܹܹܹܹܹܿ#�.�;�N�T�\�^�V�T�G�;�"��	����� �	��#�(�5�A�N�S�Z�a�a�]�Z�N�A�7�(������(�s���������~�s�q�f�a�d�f�l�s�s�s�s�s�sàìù��ùìçãàßÓÇÃÇÃÁÇÓÖà�!�"�%�!���������������!�!�!�!ƚƳ�����=�G�E�=�$����ƧƎ�u�q�s�o�rƃƚ�A�F�M�P�M�F�D�A�>�:�4�1�4�;�A�A�A�A�A�A������������������s�f�_�[�Z�a�a�s������<�U�`�d�a�e�U�I�<�#������������������<�U�b�n�{ŁŇōŐŇł�{�n�b�[�U�R�R�T�U�U���'�3�2�/�%�������޻ݻ߻������������	�������������������������������!�-�:�F�P�F�:�-�!����������-�:�C�F�P�G�F�:�:�-�-�&�%�%�-�-�-�-�-�-�Ŀѿݿ������������ݿѿĿ��������Ŀ`�m�r�u�m�h�`�T�G�;�9�;�E�G�T�[�`�`�`�`���ʾ׾����"�.�"����׾̾ʾ¾������������������������������������������������	��"�/�6�@�B�>�2�/�"�����	����������	���	������������������������������(�,�(�'���	����������������������Ľý������������������������������������ֺ̺������պȺ����������������M�f�r����������r�Y�M�@�4�*�$�"�"�'�4�M�ѿݿ����ݿѿ̿οѿѿѿѿѿѿѿѿѿ�¦¬¦ǮǳǮǭǡǔǉǐǔǡǩǮǮǮǮǮǮǮǮǮ²¿����������¿²¦¦©²²��������
��������������������������O�[�h�o�n�k�h�[�O�N�H�O�O�O�O�O�O�O�O�O K \ O 9 Z A 1  l V f Z G  +  " a e , 0 p \ D 6   ^ O j # | j R U ! . 4  R 8 S H 5 D @ _ $ S ! + D F A ? 5    �  �  �    R  ~  n  �  �  �    3  �  >    �  G  [  �  �  �  }  >  V  @    �  �  [  �  �  �  L  �  W  +  �  �  �  h  �  =    �  I  s  B  �  8  k    ~    0  ��T�����>Kƨ��o��o�D��=o=�O�;�o<�o=8Q�<#�
=� �<�=#�
<�9X=\)<���<��
=Y�>I�=C�=�P=8Q�=C�=�\)=�C�=,1=#�
=0 �=�{=�P=8Q�=��=�o=��=u=T��=@�=�E�=ix�=�9X=P�`=q��=Y�=�+=��=�9X>vȴ=��=�-=�
==>I�>�PB	_�BDB|�B)�B��B�TB-�B�BcYBV<B��B��B
��B!�fB�(BF"B�pB��B{�B,�B��B2�BڮB�B4�B��Be Bb�B�xB$V�B+�B	�xB�bB�B��B ��B�"B�[B<4B� Bh%B��B�JB�gBgB)<�B,��B6B=cA��nB��B�aB�!A��A���B	@�B��BQ@B)�BB7QB�(B<�BEB��B��B�B�=B
��B!��B)WBD�B��B@BK�BA�B�)BK�B��B��B=�BL�B0�B@gB��B$A�B?WB	��BY#B�(B��B!) B��B @�BIB��B�[B�B��B�%B=�B)?{B,��Bu*B@A��B�;B�}B�*A���A�z�A�zA竫A�>@�-A�r6?6��A���A��@��LAؙbAO:�C�oJA�'jA25�Aiq�A�Y�A~�~?��.A~
{A�$$C���A�A��BA!�-?n�A`��A��AAC3A�~�A�B�MA:�2AF)�A���A���@�@�A�40@t�	@wj|A}"zAg��AW�A��GA���A��A3P�A"J�@+c�@���A}MdA�uB��A�LA�)�AڴA��7A�}�A�Q@솾A��l?N')A�|@A�o~@���A�|�AMC�oA���A1٥Aj�A�f�A�f�?���A|��A�dC�ɠA�}A�?�A#��>���A`�/A�g|AC AAʂ�A�B��A:��AG"lA��A��P@��dA�x�@�ۏ@|	�A|�}Af��AU�A��kA�nA�u�A4��A"��@$�@��@A|�CA�~LB�A�^5A�i�Aڂp   #      �            &   G         )      O               
          r   	            *   (            4         /      &            +      *            	         �   
   
   	                  ;            %            +      =      !               !                                 7         )                                             %                                       !                  #      !                                                7         #                                                               N�RNH٘O��N�KFO	\�Nr��Oò�O�6�N_zN��`OM�HN�P x�O���O�(�O=Q\Os.�N9%�N��O��TO:(N4~�N��O�XN(�dOZ��O��^N= 7O�^N�ݞP1�N2�zO���O�O�N���O6��Nd;N[�&N�ӭOS��N�3�OZX�M��(O�:N4�N]	O��OA6Ok�zNn�LMѣ}N]J�N�0�O��N�W�  �  G    �  <  �  �  	�  �  �  �  �    �      �  a  �  u  F    �  �  p  \  �    *  �    ~  �  �  �  �  �  �    �  W  N  a  %      �  �  �  �  �  �  \  �  	6��w�+=�T���#�
�#�
;D��<�1��o;o<���;�`B=��<#�
<49X<49X<�C�<�o<�C�<���=}�<���<�/<�`B<�`B=\)=C�<��<�h=o=��=C�=\)=��=#�
=<j=49X=��=#�
=m�h=@�=P�`=H�9=L��=H�9=m�h=�o=�7L>O�=��=��w=Ƨ�=��=���>�XWX[]gtuz}}vtg[[XXXX��������������������435<BN[gotvsng[NB?64����������������������������������������������������������������� %'%%�������������!)6BDDB6)��������������������A@ADHNO[ahjklkih[ODA#/11///#UQW^������������tg[U�������������������� $)3BN[kplhig[B)���
##/883/*#
���
#/<CEEB<5/#
&#)06BFCB6-)&&&&&&&&2546;BO[ad[OB6222222,($%(/<HUagnmhYUH</,������		����������������������������LCDHNP[gjttwtrgb[PNL�wutrqrtt�����������������������������������������hehnz������������znh�����������������������)56;:6)��
#
�������������),759EZ_NB)��`gqtx�����~tmg``````}������������������}����),*.1;<5 �;BBLN[`giljg][NICB;;��������������������//4<DHQQH<3/////////��������������������[\hjt�������th[[[[[[��������������������	
#)+(###
				 #$/8>@><7/#�����������������������������	
	������).-,)zxx{�����������{zzzz��������������������''/6=>=BFGO[`bb[O6)'��������������������PHKTaelhaTPPPPPPPPPP84:<>HIIH<8888888888�

�������������������������������NOSTabmtz|~|znma^TTNHE@HTaelbaTHHHHHHHHH���
���#�-�'�#��
�������������������������
����
������������������������������)�/�3�2�*��������������������������������������������~�|������������������������������������������������������� �����	�������ݹ����������	��"�/�E�P�P�@�/�"��	���������������a�zÂÇÎÑÏÊÇ�z�n�a�U�H�D�>�?�H�U�a�ʼѼּܼ�ټּʼȼ¼üżʼʼʼʼʼʼʼ��B�O�[�c�[�Z�O�L�B�6�)�$�)�)�6�8�B�B�B�B�������׾������׾ʾ�����������������EEE#E*EEEEEEEEEEEEEEEE�N�Z�s�����������������g�Z�K�:�8�9�=�E�N����(�.�6�8�2�"��������߽�����;�G�T�`�y�������������y�`�T�G�;�4�+�0�;����)�1�/�)�)�"���� ��������������ݿ������	��������ݿοĿ������ѿݺ'�3�?�3�-�0�'������'�'�'�'�'�'�'�'�ݿ�����������ݿۿؿֿؿݿݿݿݿݿ�����������������������������������������D�D�D�D�D�D�D�D�D�D�D{DoDhDjDoDsD{D�D�D��������������ټ߼����������������������������������������������������������Žy�������������ĽĽ̽ͽĽ����������|�y�y�ܹ������ܹٹѹٹܹܹܹܹܹܹܹܹܹܿ	��"�.�;�G�U�Y�T�O�G�;�.�"������	�(�5�A�G�P�[�^�\�V�N�A�=�5�(�"����#�(�s�w�������s�f�c�f�g�q�s�s�s�s�s�s�s�sàìù��ùìçãàßÓÇÃÇÃÁÇÓÖà�!�"�%�!���������������!�!�!�!Ƴ�������<�@�=�$����ƧƎƁ�u�v�s�uƈƚƳ�A�F�M�P�M�F�D�A�>�:�4�1�4�;�A�A�A�A�A�A������������������s�f�_�[�Z�a�a�s�������#�<�U�^�b�_�U�I�<�0�#����������������n�z�{ŇŉōŇ�~�{�n�b�^�U�T�T�U�b�d�n�n����'�+�(�����������������������	���������������������������������!�-�:�F�P�F�:�-�!����������-�:�C�F�P�G�F�:�:�-�-�&�%�%�-�-�-�-�-�-�Ŀѿݿ�������� ������ݿѿǿĿ������Ŀ`�m�r�u�m�h�`�T�G�;�9�;�E�G�T�[�`�`�`�`�ʾ׾��	��"�%�"������׾Ѿʾž�����������������������������������������������"�/�5�;�?�A�=�;�1�/�"�����	��
������	���	������������������������������(�,�(�'���	����������������������Ľý������������������������������������ֺ̺������պȺ����������������@�M�Y�f�r�~���}�t�r�f�Y�M�@�8�4�3�4�;�@�ѿݿ����ݿѿ̿οѿѿѿѿѿѿѿѿѿ�¦¬¦ǮǳǮǭǡǔǉǐǔǡǩǮǮǮǮǮǮǮǮǮ²¿����������¿²¦¦©²²��������
��������������������������O�[�h�o�n�k�h�[�O�N�H�O�O�O�O�O�O�O�O�O < \ ) 6 Z A (  l V W Z 7  %   a e % , p \ D 6   [ F j # z j R W   +  R ) S H 5 F @ _ $ S  + D F A ? 5    	  �  X  �  R  ~  �  �  �  �  �  3  \    �  �  �  [  �  J  �  }  >  V  @  �  �  c  [  �  -  �  L  5    |  q  �  �  �  �  �    a  I  s  B  �  �  k    ~    0  �  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  @H  n  �  �  �  �  �  �  �  �  �  v  :  �  �  M  �  
  1  L  \  G  ?  6  -  %      
     �   �   �   �   �   �   �   �   �   �   �  
5  L  H  �  E  �  �  X  �      �    <  J  /  t    
M    �  �  �  �  �  �  �  �  �  �  �  �  �  v  Y  :    �  �  �  <  /  "      �  �  �  �  �  �  �  �  �  �  u  h  K  /    �  �  �  �  �  �  �  �  �  �  s  Y  1  	  �  �  u  *   �   �    K  k  }  �  �  w  g  S  F  8  !    �  �  b  "  �  "  C  �  	8  	y  	�  	�  	�  	�  	�  	�  	�  	j  	-  �  n  �  T  �  �  �  *  �  �  �    u  j  _  T  F  5  #      �  �  �  �  �  �  �  �  �  �  �  t  d  �  !  1  ?  L  X  _  ]  W  N  D  :  0  '  a  �  �  �  �  �  �  �  �  �  �  �  �  q    �  _  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  p  c  W  J  =  0  #  �  C  �  6  �  �          �  �  �  _    �  (  d  5   �  �  �  �  �  �  �  �  �  �  z  c  J  *    �  �  h  2  �  s        �  �  �  �  [  6    �  �  {  '    �  �  �  �  T            �  �  �  �  �  �  }  Z  3    �  �  m  :    �  �  �  �  �  �  �  �  �  �  �  �  a  5  �  �  �  f  4  �  a  K  4      �  �  �  q  Y  @  &    �  �  �  �  ~  _  A  �  {  r  j  a  X  P  G  =  3  )            �   �   �   �   �  X  i  r  t  p  l  e  X  K  8    �  �  �  z  =  �  �  e  2  +  H  �  _  �  &  C  <  )  �  �  �  *  �     �  �  	�  E  �      �  �  �  �  �  �  �  �  n  O  .    �  }    �  J  �  �  �  �  �  �  �  �  �  �  o  a  U  f  �  �  �  t  g  s  �  �  �  �  �  �  �  o  U  7    �  �  �    \  9    �  �     p  i  b  Z  Q  G  >  1  #      �  �  �  �  �  �  �  �  �  J  W  [  \  Y  Q  B  ,    �  �  �  a  )  �  �  d    �  �  k    �  �  u  [  9    �  �  {  9  �  �    �    j  �  �  �  �  �          �  �  �  �  h  .  �  �  _    �  ~  .  *  !      �  �  �  �  �  �  e  E  #     �  �  �  �  Y    �  �  �  �  �  �  y  q  b  R  >  $    �  �  ^     �   �   =  �  �  �  �  �  A  �  �  s  d  N  A  %  �  �  �    T  �  �  ~  ~  }  |  {  {  z  v  q  l  g  a  \  W  Q  L  F  A  ;  6  �  �  �  �    p  _  K  8  $  	  �  �  �  �  �  �  m  E    �  �  �  �  �  �  r  V  6    �  �  �  f  '  �  i  �     �  t  �  �  �  �  �  �  x  K    �  �  T    �  b  �  �    �  O  �  �  �  �  �  �  �  �  �  \  "  �  e  �  n  �  f  �  S  e  �  �  �  �  �  �  �  �  �  �  |  X  ,  �  �  x  @    �  �  �  �  �  �  o  �          
      �  �  �  �  �  �    o  _  N  ;  )      �  �  �  �  �  �  {  m  _  =    �  ;  g  �  �  �  �  �  �  �  x  M  	  �  C  �  P  �    (  �  W  T  Q  C  3      �  �  �  �  x  V  1    �  �  �    %  =  J  N  D  +  	  �  �  e    �  l    �    �    d  V  5  a  `  ^  \  [  Y  X  V  U  S  L  >  1  #    	   �   �   �   �      $      �  �  �  �  �  �  �  j  K  )    �  �  �  x    �  �  �  �  �  �  �  �  �  �  �  �  �  |  r  f  Z  O  C    �  �  �  �  �  r  V  D  3    �  �  �  m  E    �  �  �  �  �  �  e  G  !  �  �  �  �  e  =    �  �  \  
  �  \   �  �  �  s  X  D  H  @  +    �  �  �  �  �  ~  N    �  �  %  �  9  �  q  �  <  o  �  {  P    �    8    �  �  
     H  �  r  Y  @  '    �  �  �  �  z  V  -    �  �  ~  K  �  �  �  �  �  �  �  �  �  �  �  T  $  �  �  �  W  #  �  �  �  H  �  p  V  ;      �  �  �  t  Q  /    �  �  �  `  /  �  �  \  R  A  +      �  �  �  �  �  t  Z  D  #  �  �  P     i  �  �  �  �  ~  `  K  9  >  *    �  �  s  9  �  �  `  �  �  	6  	/  �  �    I    �  �  �  b  >    �  �  �    �  
  q
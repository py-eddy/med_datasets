CDF       
      obs    <   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�KƧ      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�w�   max       P��<      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �o   max       =�-      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @E��
=p�     	`   |   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @vh�����     	`  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @Q�           x  3<   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ϋ        max       @��           �  3�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �ě�   max       >j~�      �  4�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�3f   max       B,��      �  5�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�~�   max       B,�      �  6�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?T�   max       C��      �  7t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?]�   max       C�
^      �  8d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  9T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          5      �  :D   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          %      �  ;4   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�w�   max       P!�      �  <$   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���u��"   max       ?��G�z�      �  =   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �o   max       =�"�      �  >   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @E��
=p�     	`  >�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @vh�����     	`  HT   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @Q�           x  Q�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ϋ        max       @���          �  R,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         CW   max         CW      �  S   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��Q�`   max       ?�ߤ?��     �  T         
                     :                         
   I         A      #            	      J                     #         &   	            M         !   �         	   l            	         +N��4N.��N���Oc�M�w�NN���N}DzNŌ[P"��Oà�N���O-�N��'O���NޯwO%��O��P��<N�ON��O{:�O-��O�njO9?�N��SN� �NNO(O6{PK=�NqnoNf�uN��]O/��NJ��N��NOp�rOA��O��O���OLLN4ѰNu�N~P #O��OO�qO�g�P�%�Om�OD�WN<��P:K�O3�hN��1N7n�N�R�N� �OQ��O���o�����ě����㼃o�u�e`B�ě�;�o;�`B<t�<#�
<T��<T��<e`B<u<u<�o<�o<�C�<�t�<�t�<�t�<���<���<���<��
<�1<�j<ě�<���<���<�/<�/<�`B<�h=o=o=C�=\)=t�=�P=��=�w=,1=,1=8Q�=@�=@�=H�9=L��=P�`=T��=e`B=y�#=}�=�7L=�O�=��-=�-fbagt���������}tkgff����

�������������������������������UV[t�����������tgd[U556=@BGGDBB655555555)6:@BGB60+)���������������������{vy������������������������������������MJMR[h����������h[WM���
#/>HSRUSH</�������

���������)57BDHNTNB@5)KJMO[[abghnth[VOKKKK�������������������������������������������������!)5689865)!-DNgt�����tN5)'��������������������
),31)#/<@HOSTSI</#��������������������������������������������
##/0574/#�������������������������� ()2)����������������������������������
	��������������������������"&/;BF;/"!LNPX[fgltutpg\[NLLLLsst���������������ts��������������������MNOS[himhf[OMMMMMMMM��������������������3//08?BN[^acdb[NB=53#/<HJUURPH</
 	#/<AIF::/#
 ������� "% ���ecgiot�����������tgex{}�������xxxxxxxxxx'))/5775)��������������������������#&&!""��,,/2<HNU[_`YUOH<6/,,(,.39BIOTW`hphbOB6)(z}����������������|z�zz�����;@<)�����������������������������������	����)56>95)'#+6Bht�������[6-$������������������ 
#$'#
������

	����������������������������������
 
 �����`VUW[bmz��������zmb`������������[�g�t�u�w�y�t�k�g�[�N�H�B�A�B�E�N�W�[�[�S�_�i�e�_�S�H�F�E�@�F�L�S�S�S�S�S�S�S�SĚĦĳĽĿ��ĿĹĳĦĞĜĚĘĚĚĚĚĚĚ��������������������������������������������(�4�7�4�(������������ÓÖ×ÓÇ��z�y�u�zÇÎÓÓÓÓÓÓÓÓ�����������������������������������������zÇÓàêåàÓÇÁ�z�z�z�z�z�z�z�z�z�z��)�6�B�F�K�F�B�6�)�"��������������ʾ׾��ھʾ����M�;�4�-�-�2�A�K�Z�����������������y�o�m�`�G�;�%�"�%�B�`�m��D�D�D�D�D�D�D�D�D�D�D�D|D�D�D�D�D�D�D�D��ʾ׾����پ׾ʾ����������������������/�<�H�I�U�U�U�H�<�;�/�#���#�*�/�/�/�/�:�S�l�x�������������x�l�S�-������:�����#�/�<�G�G�>�<�3�/�&�#��������(�4�A�M�V�Z�Z�Z�Y�M�A�4�(���������������������������������y�|������������/�H�T�a�j�c�c�\�H�/���������������𽫽��Ľнݽ����������ݽнĽ������������Z�f�i�j�h�f�Z�M�C�F�M�R�Z�Z�Z�Z�Z�Z�Z�ZD�D�D�EEEEEEED�D�D�D�D�D�D�D�D�D����������������������������������������������������������������������t�����(�1�/�+�(�(��������������������"��������߹���������������'�3�4�<�3�2�/�.�'�#�������"�'�'�'����������������������������������������¿����������¿²®±«¦ ¦¨¿�nÓù����������������ù�z�a�B�@�D�V�a�n�T�a�b�k�m�p�p�m�a�\�Y�T�T�T�T�T�T�T�T�T�����Ľнսݽ޽ݽֽн̽Ľ���������������������"� �!���������	�����������������������������������y���������������{�y�u�w�y�y�y�y�y�y�y�yùú������ÿùìçàÓÇÉÓÙàìòùù�O�[�h�tčĚĤĞĎā�t�h�[�O�D�A�@�B�H�O�5�B�N�U�[�_�a�a�c�[�O�N�B�5�1�4�1�+�)�5�(�A�M�Z�s������������s�Z�4�(� ��!��(�"�.�;�G�T�`�n�l�g�\�T�G�"�	� �����	�"�Ŀѿڿݿ�������������ݿѿϿĿ������Ŀ��ĿѿؿѿпĿ��������������������������h�uƁƄƎƐƎƁ�x�u�h�f�_�e�h�h�h�h�h�h�m�x�z���������z�z�m�i�d�m�m�m�m�m�m�m�m��#�<�U�b�l�u�w�l�b�I�0������������������� �����������������������������뺗�������ɺܺ��ֺɺ��������������������(�5�=�D�Q�]�d�f�Z�N�A�(���
�����(�����������������ۼʼ��������z�t�p�s�x�������������������������������������������Ŀѿݿ����	��������ݿѿƿĿ��ÿĿĿ�����������������������������������������4�@�I�K�J�G�?�4�������׻ǻǻл�������������������������������y�x�w�u�y�����!�'�.�/�.�"�!�������������'�)�3�@�H�F�@�6�3�'�������������������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�EyE|E�E�E�E�E�E�ŇŔŠŭŹ����������ŹŭūŠřŔŎŉŇŇ¿������������������¿²¨£¤£¦²¿ 0 ` . c y v G B ! \ a F * H q + : . , � K   J ^ J _ � P u N w a  5 < Z A Z \ / T 9 ; y W 9 > G R >  B ! $ " e 8 I , $    ^  �  7  C  �  �  �  �  �    �  y  �  8  �  p  %    -  �  �  �  �  �  �  A  W    �  �  �  ,  �  h  �  �  �  �  �  O  H  �  �  �  *  �  U  �    �  a  $  z  �  u      �  M�ě���9X�e`B�o�e`B�o�#�
:�o<�t�=�o=+=C�<���<�o=8Q�<�=��<���=�9X<���<���=��=��=]/='�<���<���<�='�=Ƨ�<��<�/<��=D��=�w=L��=�+=49X=u=��=49X=�w=0 �=8Q�=��=�7L=�O�=���>j~�=�\)=�hs=q��>"��=�C�=�+=�\)=���=��=��`>�B
	BB#�B(�B
+B
YB|�B?B
u>BQ$B.�B��B90B��B]�B�4BƟBI�B��B�YB!wCB9�Bf5B�BZB:PB�B/�B�B�hB,OA�3fB�B
��B�B?�B"bB�B(�B�BǻB
B�DB��B�BM[B�B��B��B�<B|�B[�BԉB��B,��B$�TB#�JB��B�A��XB��B
?�B#A�B=�B	=BB�B��B;�B
E\BYB@YB�YB@BʠB.$BbZB�+B9�B��BxnB!��BBB@�B�gB@BaIB�SBC/B�yB��BF�A�~�B	?�B
�B%B@6B"@B�EBM�BAHB��B
6�B�	B�B ��B?�B5�B��B�WB?�B>�B?�B�UB�iB,�B$��B#��B��B��A�{"B�<A���@�L�A�0gA��2A5�*A�xnB��A�P�A�_fAG+�AhW&C��CAPs�A�%�@�z�A�YGA9	A�p�A�2=A(��A>ԁC�;�A��KA�#�A�~�?T�?��6A�YA��A˥zA�6�A'��A�sA�٘AA�!�A��8A���A?ZkAaGbA|�Ax��B�bA�o2A���A��@$�zA���@��A�ҩA}ѦAu�@���A dA�G?�q A��\C��A�u�A��CA�wP@���A�{ A�~�A6|�A�~JB&�Aʂ"A�w�AE AgLGC���AO�A�P%@�
�AA9q8A���A�#A#�A>��C�B�AЇ�A��
A��E?]�?�q�A�|"A�~TAˡ�A��A(��A��
A�w+A�Aˢ�AہeA���A@�AatA|��Awi�B��A��A�p�A�s@$AA���@��KA��yA~6Au@���A�A
��?���A�w�C�
^A��<A���   	                           :                         
   I   	      A      $            
      J                     #         '   	            M         !   �         	   l         	   
         +                              1   #            %            3               !                  -                           #                  '            5            +                                                   !   !            #            %                                                            #                              %                                 N��4N.��N6��OC'�M�w�NN���N}DzN_lGO�)TO�SNN�O�KN��'O�W�NޯwOd5N�5P�N�ON��OIyNA��O/�O.��N��SNok/NNO(O'J�O��]NqnoNf�uN��]O$��NJ��N��NOG�O$}(O��Oy5=N֏�N4ѰNu�N~O�N N���O�|O�g�P!�O]��OD�WN<��O[xSO3�hN��1N7n�N�R�N� �OQ��O���  )  �  �  }  �  B  7  f  s  &  �  �  �  �  �    |  �  �  +  �  _  >    �  t  �  �    �  m  X  p  �  }  k  N  �  �  z  �  �  r  �    �    V  h  �  -  [  
�  �    k  �  	  �  /�o������9X��t���o�u�e`B�ě�<o<�1<#�
<u<u<T��<�t�<u<�C�<�C�=�w<�C�<�t�<�/<���<�h<��
<���<�1<�1<ě�=u<���<���<�/<�`B<�`B<�h=t�=+=C�=8Q�=�P=�P=��=�w=��=H�9=H�9=@�=ě�=L��=L��=P�`=�"�=e`B=y�#=}�=�7L=�O�=��-=�E�fbagt���������}tkgff����

�������������������������������UW[gt����������tgf[U556=@BGGDBB655555555)6:@BGB60+)���������������������{vy������������������������������������XTSTV[ht��������tg_X���
#/<HMQURH</������

�����������$)15@BEIB;5)KJMO[[abghnth[VOKKKK��������������������������������������������������$)2578652)"!&0BNg����tgNB5.)"��������������������
),31) "#/<<EKOPMH</# ��������������������������������������������
 #//563/#��������������������������')-)$�������������������������������
���������������������������"&/;BF;/"!LNPX[fgltutpg\[NLLLLsst���������������ts��������������������MNOS[himhf[OMMMMMMMM��������������������10256;BNW[\_aa][NB51	
#/<HTPLH</#

	 	#/<AIF::/#
 ����� ����gdgkqt����������tgggx{}�������xxxxxxxxxx'))/5775)�����������������������������//8<DHUVWUOH<;0/////4016=BOT[\dhih[UOB64z}����������������|z���������������������������������������������	����)56>95)'4358BO[bhkkhge[OB=64������������������ 
#$'#
������

	����������������������������������
 
 �����`VUW[bmz��������zmb`������������[�g�t�u�w�y�t�k�g�[�N�H�B�A�B�E�N�W�[�[�S�_�i�e�_�S�H�F�E�@�F�L�S�S�S�S�S�S�S�SĦĳĹĿ��ĿĳĳĬĦğġĦĦĦĦĦĦĦĦ��������������������������������������������(�4�7�4�(������������ÓÖ×ÓÇ��z�y�u�zÇÎÓÓÓÓÓÓÓÓ�����������������������������������������zÇÓàêåàÓÇÁ�z�z�z�z�z�z�z�z�z�z�6�<�B�G�B�B�6�)�&�!�)�.�6�6�6�6�6�6�6�6�s���������ξ־Ҿ¾�����f�Z�S�G�F�M�Z�s��������������y�n�m�`�G�;�(�$�&�D�`�o��D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D������ʾ׾�����׾Ծʾ����������������/�<�H�I�U�U�U�H�<�;�/�#���#�*�/�/�/�/�:�F�_�x�����������x�l�_�S�-�!���!�-�:�����#�/�<�G�G�>�<�3�/�&�#�������(�4�A�M�S�Y�W�M�A�4�(�������������������������������������{����������	��"�/�;�M�T�R�H�;������������������	�����Ľнݽ����������ݽнĽ������������Z�f�i�j�h�f�Z�M�C�F�M�R�Z�Z�Z�Z�Z�Z�Z�ZD�D�D�D�EEEEEEED�D�D�D�D�D�D�D�D������������������������������������������������������������������������������������(�0�.�*�(�'��������������������"��������߹���������������'�3�6�3�0�-�-�'����������%�'�'����������������������������������������¿��������¿²®¯ª¦¢¦©¿àìù������������ìàÓ�z�n�j�i�t�zÍà�T�a�b�k�m�p�p�m�a�\�Y�T�T�T�T�T�T�T�T�T�����Ľнսݽ޽ݽֽн̽Ľ���������������������"� �!���������	�������������� ����������������������y���������������{�y�u�w�y�y�y�y�y�y�y�yùú������ÿùìçàÓÇÉÓÙàìòùù�[�h�t�āčęĚčĆā�t�h�[�O�K�D�C�M�[�B�C�N�S�[�^�`�`�Z�N�B�5�3�5�2�0�.�5�8�B�(�A�M�Z�s������������s�Z�4�(� ��!��(�.�;�G�T�a�a�[�T�G�;�.�"��	�������%�.�Ŀѿؿݿ�����������ݿѿѿĿÿ��ĿĿĿ��ĿѿؿѿпĿ��������������������������h�uƁƄƎƐƎƁ�x�u�h�f�_�e�h�h�h�h�h�h�m�x�z���������z�z�m�i�d�m�m�m�m�m�m�m�m�#�0�<�I�U�`�i�i�b�U�I�<�0�%�������#���������������������������������������������ɺֺߺɺ������������������������(�5�=�D�Q�]�d�f�Z�N�A�(���
�����(�������ʼ������ʼ������������������������������������������������������������Ŀѿݿ����	��������ݿѿƿĿ��ÿĿĿ�������������������������������������������'�4�8�7�4�.�'�����������������������������������������y�x�w�u�y�����!�'�.�/�.�"�!�������������'�)�3�@�H�F�@�6�3�'�������������������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�EyE|E�E�E�E�E�E�ŇŔŠŭŹ����������ŹŭūŠřŔŎŉŇŇ²¿��������������������¿²ª¥¦¦§² 0 ` ' Y y v G B  N c A  H o + 5 - 1 � K ! ' 5 J _ � P u Q w a  2 < Z ; X \ 2 E 9 ; y @ % < G = 8  B ' $ " e 8 I , !    ^  P  �  C  �  �  �  l  �    X  -  �  �  �  8  �  �  -  �  �  X  }  �  �  �  W  �  g  �  �  ,  Y  h  �  �  �  �  �    H  �  �    �  S  U  �  �  �  a  �  z  �  u      �    CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  CW  )  (  '  #           �  �  �  �  �  �  �  �  �  u  e  U  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  ^  @     �  �  �  �  Z  .  s  y  z  l  ^  M  ?  1  $    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  p  i  a  Y  Q  I  A  9  B  E  I  O  Z  d  _  N  =  &    �  �  �  �  �  q  T  7    7  /  '          �  �  �  �  �  �  �  �  s  S  3     �  f  _  X  Q  I  A  7  -  #    
  �  �  �  �  �  e  =     �  �    2  N  g  o  r  p  g  Y  I  8      �  �  m  ,  �  \    �  �    #  &  #      �  �  �  z  H    �  L  |  y  W  �  �  �  �  �  �  n  H    �  �  �  h    �  �  O  �  �  b  �  �  �  �  �  �  �  �  v  6  �  �  3  w  �  3  �  �    @  m  {  �  �  �  �  |  o  _  I  1    �  �  �  |  J  
  �   �  �  �  �  �  |  m  ^  P  D  8  ,       	     �   �   �   �   �  �  �  �  �  �  �  �  |  s  |  c  A    �  �  S    �  �  j      �  �  �  �  �  �  �  �  [    �  �  2  �  �  2   �   �  V  y  |  y  t  j  \  H  ,    �  �  b    �  *  �  �  L   �  �  �  �  �  �  �  �  �  �  s  a  N  7      �  �  �  �  k  >  �  �  ;  �  �  �  �  �  �  c    �  _  �  �    �  �  �  +  (  &  #          
      *  @  6  -  )  '  )  4  ?  �  �  �  �  �  �  �  t  ^  D  *    �  �  �  �  n  D     �  ?  R  ^  [  N  =  (    
�  
�  
\  
  	�  	g  �  5  D  (  �  �    l  �  �  �    '  6  >  6  #    �  �  �  �  G    �  �  �  �  �  �  �        �  �  �  �  [    �  �  (  �  k  ;  �  �  �  �  �  �  �  �  v  R  &  �  �  |  8  �  �  �  �  �  t  r  q  o  l  h  e  `  [  U  N  E  <  9  B  L  Q  B  2  #  �  �  �  �  �  �  �  �  �  �  �  �  s  [  @  %  
  �  �  �  �  �  �  x  c  B    �  �  �  ^  2    �  �  �  V  )   �   �  �    �  �  �  �  �  h  >            �  �  �  h  	  �  6    �  �  
  (  @  Y  y  �  �  �  �  T    �    ;  0  -  �  m  ^  O  A  -      �  �  �  �  �  ]  :    �  �  �    9  X  U  Q  M  I  E  A  >  :  6  5  7  9  ;  =  ?  A  C  E  G  p  e  Z  O  C  :  6  3  /  +  )  )  )  *  *  &           �  �  ~  j  U  <  #    �  �  �  �  O  	  �  k  $  �  �  �  }  b  J  :  )    �  �  �  �  �  b  ?    �  �  t  ,  �  �  k  X  @  $    �  �  �  b  /    �  �  .  �  l    �  (    "  =  L  H  ;  +    �  �  �  x  1  �  {    �  $  r  i  =  �  �  �  �  �  �  �  �  �  �  �  �  }  j  S  4    �  o   �  �  �  �  �  �  �  �  v  Z  B  '    �  �  �  n  2  �  <  �  F  T  a  m  w  z  u  g  N  1    �  �  �  j  '  �  H  �   �  �  �  �  �  �  �  �  �  �  }  h  P  6    �  �  �  �  �  t  �  �  �  �  �  �  �  �  �  �  �  �  �  s  e  W  I  ;  -    r  n  i  e  _  U  K  A  7  +        �  �  �  �  �  �  �  �  �        �  �  �  �  �  �  �  �  V  �  �  4  �  y    	�  
?  
^  
�  
�      
�  
�  
~  
0  	�  	}  	  �  �  �  x  �  1  �  &  q  �  �  �  �  �  k  E    �  �  m  0  �  �  �  �  �                      �  �  �  �  �  \  #  �  r    V  S  J  6    �  �  �  m  ?    �  �  x  9  �  �    <   �  ?  �  X  �  R  g  ]  >    �  /  �  *  �  �  �  
�  �  �  �  �  �  �  �  �  q  H    �  �  �  �  �  �  �  ]  !  �    �  -      �  �  �  �  �  �  k  N  0    �  �  e    �  J  �  [  P  D  3      �  �  �  o  E    �  �  �  R    �  �  O  %  �  �  	  	J  	�  	�  
  
P  
�  
�  
  
4  	�  	[  �    �  J  �  �  �  �  �  �  �  �  r  \  E  .    �  �  �  �  �  S     �      �  �  �  �  �  �  �  �  �  w  j  ]  Q  D  "   �   �   �  k  X  E  -    �  �  �  �  �  a  <    �  �  �  V  $   �   �  �  �  �  �  �  �  �  �  �  j  K  '  �  �  �  �  �  �  �  �  	  	  �  �  �  x  C    �  �  ?  �  �  +  Q  �  �  6  _  W  �  �  �  �  p  Y  B  (    �  �  \    �  �  C  �  �  *  d    -      �  �  �  �  �  X  !  �  �  G  �  d  �  4  �  �
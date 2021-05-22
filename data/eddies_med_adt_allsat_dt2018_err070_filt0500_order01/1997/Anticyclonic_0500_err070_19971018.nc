CDF       
      obs    6   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?����+      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N W   max       P�	�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �e`B   max       >t�      �  \   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?������   max       @EH�\)     p   4   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��\(    max       @vu\(�     p  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @/         max       @L            l  1   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�F�          �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �H�9   max       >\(�      �  2X   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�-�   max       B)�&      �  30   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       B =�   max       B)��      �  4   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?8�&   max       C�}�      �  4�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?J��   max       C�s      �  5�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  6�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          7      �  7h   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -      �  8@   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N W   max       P1�      �  9   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�����$   max       ?�_ح��V      �  9�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �]/   max       >t�      �  :�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?������   max       @EH�\)     p  ;�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�Q��    max       @vu\(�     p  D   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @3         max       @Q            l  L�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�           �  L�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D1   max         D1      �  M�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?������   max       ?�\��N;�     �  N�      	         	            f         V         +         *      	   �   	         S   )         $      8                     c   
      &      	      =      	   	         /      1   Nʾ�N��"N�N2�^N� mN�`�NW�OQ?�O��O$,�N
~�P>�NNѲ�N�2JP��O8G[O�|O��N�#7Ntw	P�	�Nf&�O�(O���PP��P��OP
O�] P�6N�xaPXa�N���N�@�O
��OX��N�2O�5O�^�N�אN�lwPE�N:yN�OصO��N WNC UNxoyNW��N��9O�V�OR��O�� N}-��e`B�����ͼ��㻣�
�o;o;D��<#�
<#�
<u<�o<�1<�9X<�j<�/<�/<�`B<�`B<�h<�h<�h=o=o=o=+=+=C�=C�=C�=\)=\)=�P=�P=#�
=#�
='�=,1=8Q�=8Q�=<j=D��=T��=aG�=aG�=u=��=�\)=�t�=�t�=��w=��T=��T>t���������������������d`egty��������tkgdd����������������������������������������IFHKN[[[egjga[RNIIIIz|��������������zzzz��������������������|{}����������������|mp���������������|tmxz|{vz������������~x��

����������������(:CC@5)���)+6=;6)()))������
�����������
+--#
��������������������������������������������������������;99BN[glmg[TNB;;;;;;����������������������5EVWO`_O5�����" #%0;<BA<0#""""""""./26<HUYaghfa]UH<6/.NJJNUaz���������znVN�������
�������������#(����������������������������}|�����������������}������������������������������ ����������)BJNNLB95)
���������������������������������'&'),6ABLOQPOLHB6,)'SVZaemqz�������zma\S&))5?5)����������������������������

�������

�����#*/4<<<;/#�����)BOTOB95*�����������������������rt���������������tr^^dgt�������������h^���������
�������������������������~�����������~~~~~~~~�#$������������������������������
####!
�����������

�����uuwy�������������zu���"/0-)���������������������������������������ĿĻĿ���������������)�*�-�)�#�����������������������������ּռּ߼���������������������������������������������U�b�n�{�ŇŇŇ�{�v�n�i�b�\�U�R�U�U�U�U�a�m�z�z�}��z�p�m�m�a�U�T�S�T�U�a�a�a�aù����������������ùðòùùùùùùùù�ܹ���������������ܹ׹׹ٹع��O�[�c�_�X�O�B�)�����������6�I�OE�E�E�E�E�FFFFF	F E�E�E�E�E�E�E�E�E���������������������������������������������0�<�n�v�}�{�r�U�<�#�
���������������@�3�+�'��������!�'�3�4�:�@�@�@�@���"�(�)�(�#���������������	���4�A�M�S�b�w�y����s�f�A���������4�"�.�;�G�T�`�m�y�}�y�m�`�T�G�;�"��	��"�ûлػܻ����ܻۻлû���������������čĚĦĳ������������ĳĚč�t�p�l�q�uāč�m�y�������������y�u�q�m�h�h�m�m�m�m�m�m�#�#�/�;�<�<�=�<�5�/�#�����#�#�#�#�#�B�[�pāĊĉā�h�O�)������������������B�ּ��������ּм˼Լּּּּּּּ��������������������������t�m�e�c�g�m�x���A�N�Z�g�s�~�����y�g�Z�A�5�(�����5�A�������������������������v�y�|�������������������������������z�U�G�N�m�z���������(�4�A�M�Z�\�`�\�Z�M�A�4�(�����"�(�(�s�����������q�f�Z�M�A�6�@�M�H�H�M�Z�s�ݿ����%�-�5�B�?�(��ݿĿ��������ĿѿݾZ�[�f�s�z�������������s�j�f�Z�Z�Y�Z�ZƎƧ���������������ƗƁ�k�Q�<�O�h�uƎ���&�*�3�*�*����	����������a�n�z�x�t�z�}�z�n�e�a�Y�V�V�X�Z�a�a�a�a�ѿݿ������������ݿѿ̿Ŀ������Ŀ˿�ŇŔŠťŭſ��������ŹŭŠŔŎŃ�}�y�{Ň�����������������������������������������g�s�s�~�����������s�o�g�Z�N�L�I�M�N�Z�gD�D�D�D�D�EEEED�D�D�D�D�D�D�D}DxDD��m�y���������z�y�m�`�T�K�T�U�`�g�m�m�m�m��"�&�.�;�G�;�7�.�$�"������������
��'�(�!���
������¿°¿����������ÇÓàåìïìàÓÇÃ�zÇÇÇÇÇÇÇÇ�s�������������������������t�s�f�f�k�s�����������������������������}�u�z�������������"�/�5�;�?�?�:�/�"��	�����������׻:�F�J�L�F�:�-�*�-�/�:�:�:�:�:�:�:�:�:�:����!�!�!�����������������[�`�g�t�t�t�g�[�[�P�S�[�[�[�[�[�[�5�B�N�[�`�]�[�N�B�5�-�5�5�5�5�5�5�5�5�5����#�����������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E~EuEvE}E�E�E������ɺֺ�����ֺʺɺº����������������úĺ��������������~�w�u�v�{��������������������������������������� U c _ W A 6 8 5 I * P / S T 8 n Z R I ; .  ) 6 D J 1 @ F N Q ; b < Q ` K p [ 1 B [ g B < ' > c _ G N Q H p  �  �  Q  j  �  �  v  �  5  _  J  7    �  �  �  e  g  �  z  �  m  �  �  �  b  =  �  �  �  �  �    ,  �  N  F  3  �  �  �  f  ?  (  �    p  �  x  �  K  	  �  ɽH�9��1��C��u;D��<#�
;�`B<�`B=�S�=+<�C�=��`=o<���=��=�P=@�=�O�=+=��>\(�=�P=m�h=aG�=�=��=Y�=aG�=�O�='�=�Q�=#�
=Y�=L��=aG�=,1=D��>bN=]/=L��=�1=Y�=u=���=�h=�C�=��=���=��
=�1>   =�;d>�> ĜB�LB	ҮB�Bg-B�:B ��B��B��B��B�B�vB�5BןB�XB#�B�B"�jB
BV-B�B?�B%��BH�B�B��B��B�B��Bn9B�Bw�B� B��B�JA�-�B�6B3�BV�B�LB�B�B!�rB�2B@Bx0B ��B)�&B��B��B�AB6�Bw�B��B�=BB�B	�FB�*B�NB�B �[B��BìB��B@;B�tB�BǔB��B$<SB��B"�B��B�`B��B8�B%�bBCB�^B@IB�lB�B��B=AB��B�8BK�B.WB��B =�B��BDB?�B<�BħBW�B"JoB��B=�BI�B E�B)��B�*B�QB�B?�BYBC�B?DA���A��%AHB�A�W�A��A�kE?8�&A��C�}�A��A꺍?���A�P1A9��Ad�@�9�A�G�Amx�A���A��AtA�z{A�I�A��^A��lA:�#AB3SA�e5AC��BiA�>AǐwA{��A���A�6�A���C��Aj�\A_otA��`AʩAG~|A�>�A�G	@}y~@^�TA���A���A��sC��@2�B@�R@��)A�e�A�z/A B?�A�H[A�1PA��?J��A�~�C�sA��,A�w?�aA�u6A8�uAb�U@��Aߛ�Am��A�EIA�t�A�aA�|�A�cA��A�o�A;�AC[�A��AD��B��A���Aǆ�A{)	A�Y\A���A��C��Ai/RA_	\A�GvA�-�AF	�A��A�~�@|�@b��A�t�A�w1A���C�d@4Dq@i@���      
   	      	            f         W         +         *      
   �   	         T   )         $      8                     d   
      '      	      >   	   	   
   	      0      1                              #         +         )                  7            -   )         '      3                     #         +                                                                           !                           #            %   %         %      -                              +                                       N<S_N��zN�N2�^N� mNt��NW�N�QVO[��O$,�N
~�O���N���N�2JO���O8G[N�s�O���N�#7Ntw	O� vN�ZO�(O���P��O�PN�"&O}=O�Nf� P1�N���N�2O
��OX��N�2O�5N�B~N�אN�lwPE�N:yN�OصO��N WNC UNxoyNW��N��9Of�OE�qO�4�N}-�  �    �  /    �  (  �  �  t  �  	�    �  c  �    �  �  ;    �  �  5  �  F    �  
  �  k  �  �  �  �    �  �  	  �  M  @    �  	�  �  �  �  �  �  
  �  T  �]/��h���ͼ��㻣�
;o;o<T��=49X<#�
<u=49X<ě�<�9X=o<�/<�<�h<�`B<�h>o<��=o=o=]/=�P=�P=��=�P=\)='�=\)=@�=�P=#�
=#�
='�=� �=8Q�=8Q�=<j=D��=T��=aG�=�+=u=��=�\)=�t�=�t�=�v�=��=�1>t���������������������fbggtw�����tmgffffff����������������������������������������IFHKN[[[egjga[RNIIII��������������������������������������������������������������������������������xz|{vz������������~x��

����������������3894)���))6:76)������
����������#('(&#
��������������������������������������������������������;99BN[glmg[TNB;;;;;;����������������������)/9BB=5)���!#+05<==<0&#!!!!!!!!./26<HUYaghfa]UH<6/.NJJNUaz���������znVN�������� ����������������$�������������������������������������������������������������������������������������� )5BIJIB53)���������������������������������'&'),6ABLOQPOLHB6,)'SVZaemqz�������zma\S&))5?5)���������������������������


 ���������

�����#*/4<<<;/#�����)BOTOB95*�����������������������rt���������������tr^^dgt�������������h^���������
	��������������������������~�����������~~~~~~~~�#$������������������������������
####!
����������

������wvwz��������������zw������,.+)���������������������������������������������������������)�)�+�)�"�������������������������ּռּ߼���������������������������������������������U�b�n�{�ŇŇŇ�{�v�n�i�b�\�U�R�U�U�U�U�a�m�q�x�z�{�z�m�a�Z�W�[�a�a�a�a�a�a�a�aù����������������ùðòùùùùùùùù�����	������� ������������6�B�O�Q�T�T�O�I�B�=�6�)�������)�6E�E�E�E�E�FFFFF	F E�E�E�E�E�E�E�E�E�������������������������������������������#�0�<�U�[�_�\�P�0�#�
�������������
���'�/�3�5�5�3�)�'��������������"�(�)�(�#���������������	���(�4�M�Z�j�n�h�Z�M�A�4�(������
��(�"�.�;�G�T�`�m�y�}�y�m�`�T�G�;�"��	��"�ûлһܻ׻лû������������������ûûû�čĚĦĳĿ����������ĳĚčā�t�s�m�vāč�m�y�������������y�u�q�m�h�h�m�m�m�m�m�m�#�#�/�;�<�<�=�<�5�/�#�����#�#�#�#�#�)�6�B�G�P�Q�M�B��������������������)��������ּռϼּټ���������������������������������t�m�e�c�g�m�x���A�N�Z�g�s�~�����y�g�Z�A�5�(�����5�A�����������������������������������������������������������������z�[�N�Y�m�z�����4�A�M�R�Z�]�Z�Y�M�A�4�(�!��(�2�4�4�4�4�Z�f�s�������������s�f�Z�M�G�S�O�O�W�Z���&�-�5�<�8�(���ݿ������ƿѿݿ����f�s�����������s�l�f�^�f�f�f�f�f�f�f�fƧ��������	�������ƟƁ�u�Y�T�Z�h�uƎƧ���&�*�3�*�*����	����������n�p�q�p�p�n�a�a�_�a�b�j�n�n�n�n�n�n�n�n�ѿݿ������������ݿѿ̿Ŀ������Ŀ˿�ŇŔŠťŭſ��������ŹŭŠŔŎŃ�}�y�{Ň�����������������������������������������g�s�s�~�����������s�o�g�Z�N�L�I�M�N�Z�gD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��m�y���������z�y�m�`�T�K�T�U�`�g�m�m�m�m��"�&�.�;�G�;�7�.�$�"������������
��'�(�!���
������¿°¿����������ÇÓàåìïìàÓÇÃ�zÇÇÇÇÇÇÇÇ�s�������������������������t�s�f�f�k�s�����������������������������}�u�z����������"�*�/�5�8�:�:�2�/�"��	�������������:�F�J�L�F�:�-�*�-�/�:�:�:�:�:�:�:�:�:�:����!�!�!�����������������[�`�g�t�t�t�g�[�[�P�S�[�[�[�[�[�[�5�B�N�[�`�]�[�N�B�5�-�5�5�5�5�5�5�5�5�5����#�����������������������E�E�E�E�E�E�E�E�E�E�E�E�E�EzE|E�E�E�E�E������ɺֺ������׺ֺɺ����������������������ºú��������������~�y�v�w�|��������������������������������� I < _ W A ? 8 < ; * P & b T - n ? W I ; 1 + ) 6 A F / 8 A 2 K ; � < Q ` K 8 [ 1 B [ g B . ' > c _ G * I A p  T  �  Q  j  �  �  v  �  �  _  J  �  �  �  �  �  �  Z  �  z  �  6  �  �      �    <  z  P  �  �  ,  �  N  F    �  �  �  f  ?  (      p  �  x  �  '  �  D  �  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  D1  �  �  �  �  �  �       *  )  $        �  �  �  �  �  �  �  �            �  �  �  �  �  �  y  G        !  +  5  �  �  �  �  �    b  F  .      �  �  �  �  �  �  �  �  r  /    �  �  �  �  w  M  $  �  �  �  �  �  �  ~  j  U  A  -        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  w  �  �  �  �  �  �  �  �  �  �  �  �  �  z  ]  6    �  �  �  (          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  5  P  h  x  �  �  �  �  �  �  �  c  6    �  �  S    �    
  
�  �  	  q  �  �  �  �  �  �  S    �  ,  
�  	y  �    =  t  i  a  _  ]  _  a  i  e  R  =  0      �  �  �  �  x  ]  �  �  �  �  �  �  �  �  �  �  �  �  �  l  T  <  %    �  �  �  �  	  	J  	}  	�  	�  	�  	�  	o  	3  �  �  Y  �  d  �  �  �  .  �  �  �  �  �    �  �  �  �  �  k  4  �  �  �  n     �   y  �  �  �  �  �  �  �  �  �  ~  k  S  ;  #     �   �   �   �   t  �    C  ]  c  ^  Q  :  %  	  �  �  {  ;  �  �  !  �  �   �  �  �  �  �  �  �  �  �  �  �  �    v  n  \  F  1        L  6  M  }  s  e  X  L  C  8  '    �  �  ^    �  ]    �  �  �  �  �  }  L    �  �  �  Y  +  �  l  �  u  �  E  �  �  �  �  �  �  �  �  �  �  z  r  j  b  Z  S  K  C  0      �  ;  6  1  '        �  �  �  �  �  �  �  l  W  C  1  ,  '  �  �  �  =  \  l  r  m  v  �    �  `  �  �  �  �  	�  S  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  k  P  5    �  �  �  �  �  �  ~  t  i  [  J  4    �  �  �  r  0  �  Q  _  5  2  +        �  �  �  �  �  |  [  4  	  �  �  f    �  �  9  e  �  �  �  �  �  c  4  �  �  ^  �  m  �    U      >  B  E  B  >  6  (    �  �  �  |  K    �  p  �  O    (  �  �    
    
    �  �  �  �  ^  2    �  r  �  L  �   �  t  �  �  �  �  �  z  h  ]  Z  \  =    �  �  v  Z  =    �  �    	    �  �  �  �  �  �  �  Y  ,  �  �  �  Y  �  N  L  c  o  {  �  v  j  ]  N  @  1  #    �  �  �  �  {  m  e  ]  0  \  j  _  L  6  5  0    �  �  �  �  b    �  &  �  �  �  �  �  �  �  �  �  �  q  ^  L  :  (        
  �  �  �  �  �  �  �      2  C  V  _  m  }  �  �  �  �  1    �  �  ~  �  �  y  l  \  L  =  ,      �  �  �  �  �  �  c  :    �  �  �  �  �  �  �  �  �  �  z  e  R  @  -          1  �        
    �  �  �  �  �  �  �  �  �  �  �  �  �  y  m  �  �  �  �  �  �  �  �  �  }  b  G  %    �  �  �  m  D      {  �  �  �  Z  �  n  �  �  �  z  	  w      
_  �  �  8  	     �  �  �  �  �  �  �  �  j  Q  5        I  {  `  D  �  �  �  �  �  �  �  u  d  R  @  /    �  �  �  �  |  U  /  M  J  ?  %    �  �  �  �  U    �  x    �  �  �  3  �  Z  @  <  7  2  ,  #        �  �  �  �  �  �  �  �  y  g  U        �  �  �  �  �  �  �  �  l  W  B  ,      �      �  �  �  �  �  {  g  O  0    �  �  t  4  �  �  H  �  s    	m  	�  	�  	�  	�  	�  	�  	�  	i  	0  �  �  I  �  U  �    A  >    �  �  �  �  �      	        �  �  �  �  r  C    �  h  �  }  p  e  Z  O  D  9  -  !    	  �  �  �  �  �  �  �  �  �  �  �  �  }  e  K  1    �  �  �  �  �  �  c  A    �  �  �  n  P  0    �  �  �  n  ?     �  c  1     �  �  �  �  o  �  �  �  �  �  l  C    �  �  �  Z    �  �  l  :    �  �  	�  	r  	d  	o  	�  
  
  

  	�  	�  	j  	   �  i  �  C  �  �  �    �  �  �  �  �  �  p  G    �  �  �  @  �  �  C  �  �  =  �  %  T  S  E  0    �  �  f    �  x  %  �  f  �  U  �  �      �  �  �  �  �  �  s  b  R  I  <  /  !      �  �  �  �
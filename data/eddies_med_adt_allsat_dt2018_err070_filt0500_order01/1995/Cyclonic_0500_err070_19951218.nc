CDF       
      obs    F   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?���Q�       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�w�   max       P���       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       <�o       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>Q��R   max       @F�          
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @v~�\(��     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0�        max       @N�           �  6�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�&        max       @���           7`   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �%   max       <e`B       8x   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�EE   max       B4�)       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��1   max       B4�o       :�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >觇   max       C��l       ;�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�+�   max       C��J       <�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          G       =�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =       ?   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7       @    
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�w�   max       P~�       A8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�9XbM�   max       ?��u%F       BP   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �Ƨ�   max       <�o       Ch   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>Q��R   max       @F�          
�  D�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @v~�\(��     
�  Op   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @          max       @N�           �  Z`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�&        max       @��            Z�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A�   max         A�       \   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��	� �   max       ?�S&��     �  ]            F   
   !                  	                        	   ,   :                        	         8   9            8                        #                     :                     1         +         
      #       N�3N:�N2�PA��N��On�lN}��N�f\NG�N8�O=%�N��TO�JO9�OZ�O1��P&��N�ιN��XN���P���PO~:N�>�N2`&N���OioKO1W�N�cdN�'N�]�NAٰO��yPn��P^��N�O"�OP�pP
qxO`�N� �O�<O�J�P$PO&N�O���O�c�N_tO�"O�#�N�ݺOU��O��P'�*N�s�O�1�O���O�O
u�O� ]P3|AN�M�w�O֓�Oc#O��N��O+�7O�-9N��{M�$<�o<e`B<D��<o;ě���o��o���
���
�ě���`B��`B�o�#�
�#�
�D���e`B�u��o���
��1��1��j��j�ě����ͼ��ͼ�����`B��h��h��h��h�o�o�o�+�C��C��C��\)�\)��P��P��w�''8Q�<j�<j�@��Y��Y��q���y�#�y�#�y�#�}󶽋C���C���hs���������P���P���-���
��Q������

�������������������������������������������������

������V[chikjh\[[VVVVVVVVV��������������������IOR[_a`^\[ZUOMHCIIII��������������������


������������������������������������������������������������������������������������������>BO[hstwpurh[NLLA98>y�������������{uuvyy�������������������ht������������th\_h#/<DHIHE=<9/%#������������������������������������������
#0bm������nR0
��z�����������������xz��������������������������������������#&/4<CHA<2/# HTamz}�����~maTRLHHH25BJNUYbgmg[UNLB;522MNT[\gggg[NLDBMMMMMM66COPWOC656666666666jmz|��������zvmlifjj����������������������������������������)6O[t����ym[OCI=,'$)BM[t�������pg[KD0,/B����������������������������������������!#/<HU_efdaUHG3/+##!������������������������������������������������JUaajnz~���unaXULNIJ����!������������������������������������������������������������������������ 
������������FHNUV[YUHFFFFFFFFFFF�������������������������������������������������������t��������������zqqtt~��������������|zy~����������������������������������������SU[bgt����������t[NS��������������������knz�������zynlgedkk')*35<BNO[][[UNB5-)'367:=BN[gt}}umgcNB53[et�����������to^Z[��������������������HHKUUVUHGEHHHHHHHHHHUU\nz������������a[U#0<>FIJLIF<00#9<>AIJUWUXVRI<732379bn{|������~{yvojba`b_alnz����������nca__�������������
###$'# 
+/1<>?</+'++++++++++��������������������������������������������������������������������������������D�D�D�D�D�D�EEED�D�D�D�D�D�D�D�D�D�D߼Y�M�4�����4�J�P�M�f�r�|�}������f�Y�e�[�Y�Q�Y�e�r�w�~��~�r�e�e�e�e�e�e�e�e������%�<�D�H�K�D�F�M�M�K�H�<�/�#���|��������������¼����������������	����	��"�#�%�"���	�	�	�	�	�	�	�	�}�|����������������������������������������ÒÌÉÇ�y�zÇÓàìù����������ùìàÒ�	��������������	�������	�	�	�	�Ŀ������¿Ŀѿֿݿ����������ݿѿ��U�R�W�a�i�n�p�zÇÓàåäàÓÇ�z�n�a�U�s�h�f�d�f�m�s���������������������s�s���m�_�Z�W�`�m�y�}��������������������������پ־������;�N�S�R�T�G�=�8�.����6�1�*�.�2�6�?�B�E�O�[�]�[�S�O�N�F�B�6�6�H�G�C�H�J�U�V�a�n�z�~�z�z�{�z�o�n�a�U�H���ݿؿҿݿ��������'������������������q�g�G�A�(�N�g�����������������˿����y�h�b�`�d�����Ŀѿܿ�������Ŀ����	��������	��"�'�.�/�.�$�"����ìçàÖàäìòù��ùðìììììììì�H�<�<�9�<�?�H�U�a�f�n�q�r�n�a�U�H�H�H�H���
�����"�/�;�<�H�P�S�K�H�;�/�"�������������������������������y�v�y�����������������������y�y�y�y�y�y����������	����	�����������������������������������������������������������N�L�K�N�V�Z�g�o�n�g�Z�P�N�N�N�N�N�N�N�N������ĿĳĦďĕĚĦĳ������������������a�a�m�l�a�e����������C�K�H��������s�a���������ʾ׾�	��"�:�G�[�e�G�;�"�	�𾫾�ھ׾־о׾ھ�����������������6�*�%�$�*�4�C�\�h�uƁƁ�u�r�h�e�\�O�C�6�N�M�K�E�F�M�Z�g�s�������������{�s�g�Z�N�/�"�����������	��/�;�H�T�^�j�e�T�;�/�s�m�g�d�d�g�s�������������������������s�x�x�m�_�X�S�K�S�_�_�l�x�}�������������x�����������������ÿĿſѿԿ׿ѿͿĿ������ʼüɼԼ����!�.�2�<�=�:�.����ּ�ƧƊ�u�h�fƁƇƚ�����������������Ƨ�6�.�*�!� ���*�6�C�C�H�S�\�h�\�O�K�C�6�@�6�3�&�'�3�@�L�e�r�~�����������q�Y�L�@�^�i�z�����������������������������z�m�^�M�D�A�@�A�M�Z�c�e�Z�M�M�M�M�M�M�M�M�M�M��������)�6�B�C�O�[�[�O�O�B�6�)��������޻�������"�'�3�3�.�'�������������������ûлջۻлȻûû�������œŌ�x�n�m�l�{ŔśŠŭŶ��������ŹŭŠœ������������$�0�7�@�C�=�4�0�$���������r�e�N�Y�^�r����������	��ɺ����t�s�h�a�b�h�tāĉĎĚĦĭĳĽĦĚčā�t�
��������ĸĳĸĿ������������	��"��
��
���������������
���#�-�:�B�B�<�0����������ƾʾ׾���� ��������׾ʾ���������ùóìàêìõù�������������������z�n�a�U�H�?�3�-�+�&�<�H�U�a�j�nÂÅÁ�z��¦��t�^�c�t¦�������
���
�����˺ֺֺӺֺۺ�����ֺֺֺֺֺֺֺֺֺ�F$F#F$F.F1F=FGFAF=F1F$F$F$F$F$F$F$F$F$F$�ù����������������ùϹ���!����ܹý����������������������½ĽɽĽĽ������������ݽɽн������(�9�=�4�(������9�3�8�:�F�O�S�_�l�z�������x�l�_�T�S�F�9�m�e�T�R�O�T�a�p�z�������������������z�m������!�.�:�S�v���y�r�l�`�S�G�:�.�ECE9E<ECEOEPE\EiEuE|EuElEiEaE\EPECECECECE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� H y b D H _ g M r 2 = { ! P ( r + R g W H > S I $ ( @ E G Z ` i e 2 - \ 3 0 - i U P a B N _ I B F L \ ? T q C ; 5 4 Q J f _ 4 3 k \ Q < d I  *  �  }  �  8  \  �  �  G    �  8    �  A    �    4  6  j  �  �  _  �  �  �  �  J    z  Y  �  �  �  �  �  y  �  S  E  !  9  �  :  2  !  b  @    �  d  W  >  �  y  ;  9  )  j  Z      3  �  2  �  O  �  '<e`B<49X;�o�}󶻃o�t��e`B�#�
�#�
�o��h��o�ě��������ͼ������/��h��h��o������h��/��w�'�P�+�o���+�H�9��{��E���w�8Q�Y���E��@��8Q�D���q����%�D����7L����@��ixս���e`B�aG���+��S����P���-���㽟�w������ Ž���P�����h��j������-��^5�   �%���`B$~�B+�B�TB!KkB�]B?B�A�EEB��B4�)B��B!�B��B�;B%�BĳB�VB)�B!��BQUB&��B+Bx�B��B/�A��B<5B��B0�VA��LB�'B�B LB	&WB�B�B%B�oB��B!��Bw�B-J�B�BQLB"�B�5B�_B��Bc�B׭B
�@B
�B�(B�DB	�B8�BH?B�(B�^B
�MB"W�Bn�B�"B%�LB&�kB(��BJ�B=cB�B[�B$��B��B@�B! FBAUB��B�A��1B��B4�oB�kB!@?B��B�HB>�B��B BA�B"fB?dB&�wB*ΧBPB�B?
A�q�B>�B[IB0�SA���B��B<cB�CB	@B@`BB�BE�B�(B�8B";DBb�B.>=BE�B?�B"D�B>�B��BF�B@�B��B?zB
�xB?�B��B	��B��BA�B�mB�:B
?B"Q�BE�B�B%�GB&��B(��BB�PB?�B@@�2�AЍ�C�;p@�[?�Z�A�@�jA���A�zAK+�A�xA��A~"�A�+ AF�nAoIgA]mA�V�AƒNA�مA�kcAw�A\��A�t�AŇWA�kCA���ApuAZG�A�}oA��<A���A��#AZ:�AU��BWA���A�;A�uy@���Aw#�AU�B3�B �A?�g)A���A=ҚA���@���@�V�A�4�B	��@%�jA���A�)=A��!AS�?AΉRAŭ�A��@A�?C��l>觇A!��A2��@�w?A�nA�{C��ZC�@���A�$C�G�@�F?�bnA�uY@��A�s�A�m�AK�YA�v�A���A{�A�x AEYAm-%A^��A؀�A��A�t�A���At+�A] �A��gA�+A���A�XTAp�YAZY�A�L~A���A���A�|AZ��AUʔBA A��OA�>A�w@��QAxA
�aB�B ?�?�w�A���A=%�Aց�@��@�R�A�w�B	�%@3�_A��A��A�P�AS^uA΀	A�Z�A�u@CqC��J>�+�A!�bA38�@���A��=A�C�� C��         	   G      !                  
                        
   ,   ;                        
         9   :            8                        $               	      ;                     2         ,               $   !   	            5                                       +            =   5                                 =   3            '            %   +         '                     /      #   !            -         %                                 '                                       %            7   3                                 /                           %   %                              -      #   !            -         %                     N�3N:�N2�P�WN��O\�QN}��N�f\NG�N8�N��N��TO�JN�OZ�O1��PpN�ιN��XN���P~�PFb�N�>�N2`&N���OQ�O1W�N�cdN�'N�]�NAٰO3�P(XO�H�N�N�NO0؇O�c�O`�N� �O�<O�J�O��EO&N�Oe�O�(�N_tOӕO�#�N���OU��O�BP��N��9O�1�O��?O�O
u�O� ]P({�N�M�w�O֓�N��]N�N��O�\O�-9Ne��M�$  �  �  f  �  ;    s  �  �  �    �    �  �  <  �  d  ,  G  T  N  �  ]  �  �  �  �  �  �  j  �  �  u  �  �  :  �  +  4    �  �  u  "  �  �  V  W    �  �  �  -  �  �  �  �  f  �  �  �    |  d    �  �  �  �<�o<e`B<D�����
;ě����
��o���
���
�ě��e`B��`B�o��C��#�
�D����o�u��o���
��9X��j��j��j��/�������ͼ�����`B��h��h�+���m�h�o�\)�\)�H�9�C��C��\)�\)�#�
��P�@��@��'<j�<j�H�9�@��]/�aG��u�y�#�}�y�#�}󶽋C���\)��hs�������������-���-���T��Q�Ƨ���

������������������������������������������������������������V[chikjh\[[VVVVVVVVV��������������������IOR[_a`^\[ZUOMHCIIII��������������������


������������������������������������������������������������������������������������������=BOT[][VONB@========y�������������{uuvyy�������������������ejt�������������thde#/<DHIHE=<9/%#����������������������������������������#0bo{�����nP0
� {�����������������y{��������������������������������������#/<=C=</# JTamwz�����|maTRMIIJ25BJNUYbgmg[UNLB;522MNT[\gggg[NLDBMMMMMM66COPWOC656666666666jmz|��������zvmlifjj����������������������������������������(+6O[t�����wph[RB0*(AEN[gt~������tg[NA@A����������������������������������������'/0<HSU]acda`UH7/-%'��������������������������������������������������JUaajnz~���unaXULNIJ����!���������������������������������������������������������������������������	�����������FHNUV[YUHFFFFFFFFFFF�������������������������������������������������������t��������������zqqtt���������������}{z����������������������������������������SU[bgt����������t[NS��������������������knz�������zynlgedkk')*35<BNO[][[UNB5-)'367:=BN[gt}}umgcNB53\dft�����������tg`[\��������������������HHKUUVUHGEHHHHHHHHHHUU\nz������������a[U#08<@B><0#!;<CHIRRUVTQJI<84459;bn{|������~{yvojba`b_aemnz��������znea__�������������
"!
	+/1<>?</+'++++++++++��������������������������������������������������������������������������������D�D�D�D�D�D�EEED�D�D�D�D�D�D�D�D�D�D߼Y�M�@�'�"�$�$�-�@�M�f�r����������q�f�Y�e�[�Y�Q�Y�e�r�w�~��~�r�e�e�e�e�e�e�e�e������#�&�/�<�C�D�H�L�L�J�H�<�/�#���|��������������¼����������������	����	��"�#�%�"���	�	�	�	�	�	�	�	�}�|����������������������������������������àÛÓÑÐÓÔàìùù��üùíìàààà�	��������������	�������	�	�	�	�Ŀ������¿Ŀѿֿݿ����������ݿѿ��a�Y�]�a�l�n�x�z�{�z�q�n�a�a�a�a�a�a�a�a�s�h�f�d�f�m�s���������������������s�s���m�_�Z�W�`�m�y�}�����������������������	����ܾھ�������;�J�P�Q�E�8�.�"�	�6�1�*�.�2�6�?�B�E�O�[�]�[�S�O�N�F�B�6�6�H�G�C�H�J�U�V�a�n�z�~�z�z�{�z�o�n�a�U�H���ݿؿҿݿ��������'��������������q�h�H�A�6�/�g���������������������������y�i�c�a�g�����Ŀѿۿ޿������Ŀ����	��������	��"�'�.�/�.�$�"����ìçàÖàäìòù��ùðìììììììì�H�@�?�G�H�U�a�b�l�l�a�U�H�H�H�H�H�H�H�H�������!�"�/�8�;�H�O�Q�J�H�;�/�"�������������������������������y�v�y�����������������������y�y�y�y�y�y����������	����	�����������������������������������������������������������N�L�K�N�V�Z�g�o�n�g�Z�P�N�N�N�N�N�N�N�N������ĿĦėĚĜĦĳĿ���������������������s�j�q�s�j�m�|�������������������������߾پܾݾ�����	�� �*�/�-�&��	����ھ׾־о׾ھ�����������������6�+�*�(�*�6�<�C�O�\�_�h�n�h�a�\�O�C�6�6�Z�O�N�H�H�N�O�Z�c�g�s�~���������x�s�g�Z�/�"���
���"�/�;�H�T�^�b�a�[�T�H�;�/�s�m�g�d�d�g�s�������������������������s�x�x�m�_�X�S�K�S�_�_�l�x�}�������������x�����������������ÿĿſѿԿ׿ѿͿĿ������ʼüɼԼ����!�.�2�<�=�:�.����ּ���ƧƎ�{�|ƁƍƚƧ�������������������6�.�*�!� ���*�6�C�C�H�S�\�h�\�O�K�C�6�@�>�5�5�@�J�L�Y�e�r�����~�}�r�f�e�Y�L�@�m�k�m�z�����������������������������z�m�M�D�A�@�A�M�Z�c�e�Z�M�M�M�M�M�M�M�M�M�M��������)�6�A�B�M�M�B�6�)�����������޻�������"�'�3�3�.�'�����������������������ûлллĻû�������œŌ�x�n�m�l�{ŔśŠŭŶ��������ŹŭŠœ�����������$�0�6�=�=�B�=�2�0�$�����������r�e�R�]�_�r�~�����λ����ɺ����t�t�h�b�c�h�s�tāčĚĦĪĦĥĚčā�t�t�
��������ĸĳĸĿ������������	��"��
��
���������������
���#�+�9�@�A�<�0����������ƾʾ׾���� ��������׾ʾ���������ùóìàêìõù�������������������z�n�a�U�H�?�3�-�+�&�<�H�U�a�j�nÂÅÁ�z��²¦�t�b�f²�������
�
������˺ֺֺӺֺۺ�����ֺֺֺֺֺֺֺֺֺ�F$F#F$F.F1F=FGFAF=F1F$F$F$F$F$F$F$F$F$F$�ù����������������ùϹ���!����ܹý��������������������������������������������ݽڽ������(�4�7�;�4�(������9�3�8�:�F�O�S�_�l�z�������x�l�_�T�S�F�9�m�g�a�T�S�P�T�^�a�m�r�z�������������z�m������!�.�:�S�v���y�r�l�`�S�G�:�.�ECE:E=ECEPE\EiEiEiE_E\EPECECECECECECECECE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� H y b - H ? g M r 2 . { ! Q ( r ) R g W H ; S I & % @ E G Z ` ` P  - A )  - i U P Z B = P I > F R \ E U k C < 5 4 Q L f _ 4 " a \ < < ; I  *  �  }  W  8  �  �  �  G    �  8    R  A    �    4  6  Y  {  �  _  �  �  �  �  J    z  �  ?  )  �    }  [  �  S  E  !  �  �  L  c  !    @  �  �  4    �  �  \  ;  9  )    Z      �  D  2  W  O  ~  '  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  �  �  �  �  �  �  �  �  �  �  �  }  c  I  0     �   �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  f  \  S  G  ;  .       
    �  �  �  �  �  �  ^  $  �  �  W  �  �  p  �  ~  W  F  7  ,            �  �    j  K  ;  ;  ;  5  /  (  $  %  (  ,  1  7  @  L  ^  t  �  �  �  �  �  �  �  �  }  �  �  s  ?    �  l    �  p    �  6  �  #  s  m  e  R  >  -    �  �  �  �  s  �  �  �  n  >    �  �  �  �  �  �  |  m  ^  O  ?  +      �  �  �  �  �  w  \  @  �  �  �  �  �  �  �  �  �  �  o  [  C  (    �  �  �  �  w  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  3  �  �  �  �      
    �  �  �  W    �  i    �  c  !  �  �  �  �  �  {  o  `  O  :  #  	  �  �  �  �  `  '  �  �      
  �  �  �  �  �  �  �  �  �  h  H  $  �  �  �   �   �  c  c  b  f  e  X  G  3  �       �  �  �  �  <  �  �  3  �  �  �  �  u  e  U  D  1      �  �  �  �  k  H    �  4   �  <  1  %     )  2  3  .  )        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  V  <    �  �  �  s  6   �  d  [  L  4    �  �  �  �  e  @    �  �  �  o  0  �  �  ?  ,  )  %         �  �  �  �  �  Y  8  .  i  �  �  �  �  �  G  E  C  4  $       �  �  �  �  �  �  n  \  B    �  �  �  =  =    �  �  �  e  .  �  �  o  =  	  �  �  s  >  �  |   �  K  M  C  2       �  �  �  `  (  �  �  Y    �  /  �  
   �  �  �  �  �  �  �  o  ]  L  :  )    
  �  �  �  �  �  �  u  ]  W  R  L  F  ?  7  /  '        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  f  J  ,  
  �  �  �  �  �  �  �  �  v  b  O  ;  %    �  �  �  �  ]    �  T  �  �  �  �  �  �  �  �  �  v  f  U  H  ;  /    �  �  n  �  �  �  �  r  [  C  )    �  �  �  �  }  a  G  .    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  o  c  W  K  ?  �  �  �  �  �  �  �  �  o  R  0    �  �  S    �  �  s  K  j  X  E  3       �  �  �  �  �  �  t  ]  F  /       �   �  \  n  �  �  �  �  �  �  k  W  C  K  Z  L  1    �  �  }  -  J  U  h  �  �  }  o  ^  F  .      �  �  I  �  �  "  ]  w  P  a  �  �  �  '  M  e  r  s  f  P  .     �  u     �  �   �  �  �  �  �  �  �  �  �  }  k  Y  E  (    �  �  �  m  C    �  �  �  �  �  �  �    `  >    �  �  �  �  U  !  �  �  }  "  3  9  6  -    
  �  �  �  �  Y  .  �  �  �  _  0  �  �  �    #  �  �  �  �  �  �  r  7  �  �  O  �  K  �  �  2  E  +  (  $      �  �  �  �  �  �  �  s  T  )  �  �  �  r   �  4  &    	  �  �  �  �  �  �  �  �  q  S  *  �  �  P    �        �  �  �  �  �  �  �  }  \  6    �  �  v  H        �  �  �  i  >    �  �  �  g  =    �  �  �  Q    �  d  0  �  �  �  �  �  �  �  d  D  "  �  �  �  t  5  �  �    u   �  u  f  W  J  9  &  
  �  �  �  �  �  {  [  :    �  |     �            !         �  �  �  e    �  y  0    ;  J  �  �  �  �  �  �  �  �  �  �  v  A  	  �  �  D  �  }  �  �  �  �  �  �  n  U  <    �  �  �  �  `  5  	  �  �    O     N  S  U  R  O  L  G  C  >  4  &       �  �  �  �  �  (    W  Q  4    �  �  �  �  �  �  �  n  S  '  �  �  �  I    �  �  �  �  �  �      9  C  1      �  �  �  Z    �  �  P  �  �  �  �  �  �  x  Y  6    �  �  �  �  �  h  4     �   �  �  �  �  �  �  �  �  �  �  q  W  9    �  �  g    �  H   �  �  �  �  �  �  �  �  U    �  �  G  �  e  �  ^  �       �  �      	  �  �  �  �    F  	  �  �  T    �  �  R    �  �  z  ]  C  2  *  (        �  �  �  �  �  |  D    �    �  �  �  �  �  �  |  o  ^  I  .    �  �  �  w  >    �  ]  �  �  �  �  �  �  h  L  .    �  �  �  r  E    �  �  �  �  �  �  �  �  �  �  ^  2    �  �  �  �  t  Z  @  $    �  �  f  b  Z  M  7      �  �  �  �  �  z  U  -  �  �  |  1  �  �  �  �  �  �  �  t  K    �  �  �  `  0  �  �    '  a  �  �  �  �  �  �  �  �  �  ~  s  i  ^  S  H  ;  /  #      �  �  �  �  �  �  �  �  �  �  �  �  �  �    u  k  a  W  M  C    �  �    F    �  �  `  #  �  �  <  �  g  �  �    m  u  o  i  e  l  u  z  z  v  o  h  Y  B  %    �  �  �  w  Q    .  a  d  d  _  Z  E  (    �  �  |  C  	  �      }  �      �  �  �  �  �  y  a  I  4  #      �  �  �  �  �  q  S  �  �  �  �  �  �  �  �  v  W  1  
  �  �  �  a  ,  �  �  �  �  �  �  �  �  l  E    �  �  |  F    �  n    �  +     #  
F  
K  �  y  H    
�  
�  
^  
  	�  	T  �  	  I  �  �  �  �    �  k  U  ?  (    �  �  �  �  �  �  y  a  I  /    �  �  �
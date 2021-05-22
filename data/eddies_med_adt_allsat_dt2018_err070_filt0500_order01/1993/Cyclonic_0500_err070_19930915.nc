CDF       
      obs    G   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�/��v�       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�   max       PC��       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��`B   max       =o       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��R   max       @F~�Q�       !    effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ۅ�Q�    max       @vZz�G�       ,   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0�        max       @Q`           �  70   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�           7�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ����   max       <���       8�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�i�   max       B0�7       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�O�   max       B0@�       ;   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =�h�   max       C���       <0   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >'ܴ   max       C���       =L   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          9       >h   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          3       ?�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3       @�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�   max       PC��       A�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���Mj   max       ?�:�~���       B�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��`B   max       =o       C�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��R   max       @FAG�z�       E   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��Q��    max       @vZz�G�       P(   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0�        max       @Q`           �  [@   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�q        max       @�           [�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >[   max         >[       \�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���҈�   max       ?�:�~���     0  ^         
            %            	            &      	         -                                             +                                             9   8   !      %   	         !         &      
      	                  
N�vO�bN���ON�NT�N�	P8�N��LN� N��NE�$OLJ�P�N��]P��O��OP�EP�&N<�P��N,NYN
6WOM�rN���N�tN�~:M�N�w�P 6OqR\N���N��O
�zP+
O�ɢO�!�M��:N�TaN�
CO��N��O/Oy�O�N�{O��
N���O��O��P!oP.�HO���O=��O� fN���N�2�Ox��O�d-O1�N@_PC��NIN&N���OU�|No��N�X7O)@'NF�N��N���N(��=o<��<�C�<u;�`B;ě�:�o%   �D�����
���
���
�ě��ě��ě��o�o�o�t��D���e`B�e`B��t����㼬1���ͼ���������/��/�������o�+�C��t���P����w��w�''49X�8Q�8Q�8Q�8Q�8Q�@��@��D���D���Y��Y��ixսixսm�h�u�u��������C���\)��hs�������㽛��ě��Ƨ��`B������������������������������������������������������������������

��������gmvz}���zmgagggggggg,/3;?;84/**,,,,,,,,,�)'*5N[sxzwr]NB)��#/<?HLKH</'# ����������������������������������������7BOOX[\[OHB777777777|������������������|w���������������~vuw��������������������0CO\u���h\C6*GJTam������zma\TMHFGMU[gt�������tg[NIFLMt{z~�����������wuomt������������������������'.,#
�������������������������������������������������������269BCO[\htqhh[OHB622Vbnw{~{qnmb^VVVVVVV`aemzz|{|zzpmga__]``����������������������������������������9>Pmz��������zaH==89`g�����������tpgc\[`Y[fhituvutkh][VUYYYY*/;ACFC?;/+'$#******#$/;<=<::64/#�
#0<Un{���{U<#
���������������������������
������������������������������������������������cgit�������tigcccccc���
#.11-(
�����#/9<=<<50/#)36ABKMGB61)�������������������������������������������������������������������	�������:BDN[e\[NNB7::::::::z��������������}trnzdnz�����������zungddO[g�����������g[LEFO���")+&���������//<HUantwz{zaUHB<90/M[hp}����~}tnh^YRKMxz���������������zwx��������������������ggrt�����������~tggg�����������������������������������;<CHUaa`ZWUSIH<8558;������������������������);������������������������������|ttppt�����������/5;NVkmnkg[ND@A@<53/�� �����������������������������!#%0<AINRRKH@<0-#"!!-038<@A<60.))+------��������������������#/<HLOKH<5/(#:<GHIRTLHE<;::::::::��u�s�t����������������������������H�G�<�/�#���� �
��#�)�/�<�D�H�N�I�H�4�*�(�����(�4�A�B�M�Z�\�Z�T�M�A�4�4�s�j�f�Z�^�]�a�f�s�������������������s�)�!�)�/�6�B�B�F�F�K�B�6�)�)�)�)�)�)�)�)��������������������������������������ؿѿ��������y�z�~�������ѿݿ����� ������$���"�$�*�0�=�A�F�F�=�5�0�$�$�$�$�$�$������)�*�+�)��������������������(�*�5�;�?�8�5�-�(����a�Z�a�a�n�y�z�Â�z�u�n�a�a�a�a�a�a�a�a�y�m�`�T�;�.�"��"�*�.�;�G�T�`�b�m�y�|�y�������������5�C�O�\�r�v�h�Y�O�C�6��g�]�Z�N�A�7�7�A�N�Z�]�g�s��������s�g�g��ʾ����������ʾ���8�A�F�F�A�;�"�	������ƶƴ��������������%�%�$�����������������������������������	������s�f�Z�A�6�M�S�s�������׾������׾����s�g�b�b�e�g�n�s�}�|�s�g�g�g�g�g�g�g�g�g�g�����s�h�s����������������������������"���"�/�;�<�;�;�/�"�"�"�"�"�"�"�"�"�"�������������	��	����������������������Ϲ������������ùϹܹ����������ܹ��
��������������������������
���
�
�!���!�%�-�:�A�F�K�F�?�:�-�!�!�!�!�!�!���������������	��"�#�/�7�/�*�"��	��������������������������������������������¦­²º¹³²¦��������޿������5�A�Q�W�N�A�5�������������������)�,�*�!�������꽒��������������������������������������s�o�l�s���������������������s�s�s�s�s�s�N�L�B�7�5�+�5�7�B�G�N�[�g�t�{�~�t�g�[�N�������x�c�`�`�]�V�U�Z�s�����������������������(�5�A�N�Z�c�b�Z�G�A�5�(�����������úɺֺ�����������ֺɺ�������������������������� ����������������àßÔÓËÓàìñùý��ùìàààààà�	������������	��� ����	�	�	�	�	�	��ùìàÓÌÑàì����������������������H�?�>�H�P�U�a�a�n�z�}Â�z�n�a�U�H�H�H�H�Ŀ����������������Ŀѿڿ�����ݿѿĿy�v�n�m�w�������������ɿпͿĿ��������y�"�������"�(�/�;�F�H�O�N�H�;�/�"�"������������� �������������������������.�"��	����׾�	��.�;�T�W�c�`�T�;�.����������������������������������������ŠŕŅ�}ńŇŖŠŭŹż��������������ŹŠ���������������Ŀƿѿֿۿ׿տӿѿͿĿ���¿¬¦¿�����������
����¿�����������!�:�`�y���������d�X�;�!���������������������Ϲܹ���޹̹ù��������ù������ù͹Ϲܹ������������ܹϹú����~�t�e�`�N�^�e�~�������ºϺƺ�������������$�'�3�@�I�@�?�3�'�"�����ùøìëáàÚÖàìù������������ùùù����Ķİ���������
�������
�������̼�u�z�~���������������㼽������׾վʾþþƾ˾׾������	�
��	�����׺3�/�3�<�@�I�L�O�Y�\�Y�O�L�@�3�3�3�3�3�3ùÛÐËÝë�����)�B�J�>�2�.� �����ù�ܻջлû������ûлܻ���������ܻܻܻ������������������������������������������h�f�tĀčĦĳĿ����������ĿĳĚčā�t�h�������������������������������������������$�%�$��������彫�������������������Ľнݽ��ݽʽĽ�����������(�0�4�8�4�(������������������������������������������������ECE?EAE=E=ECEPE\EbEcE\EVEPEJECECECECECECD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� ; f I 7 Q y : V O ( ] K + x Y 2 A k V J ? Z M � A ^ { % ] ( * G Q S . g � ` J ? C , ! D . d j ' n > 1 ^ @ V n W N t Y V F � i { F D / d B x 3      �  �  �  �  l    �  ,  �  �  �  O    /  �  �  �  f  �  B  9  �  (  �    G  	  �  �  �  �  :  Z  $  �  w  �  �    �  O  �  #  *  �  �    �  �  !  T  �  =  �         ;  �  �  �  �  _  |  0  l  �  �  �  D<���<ě�<o;�o%   ;�o�t��#�
�t��49X�T����o��/�T���49X��`B��C����ͼ49X�e`B��C���C��,1��j��P�o���8Q�P�`�,1�t���w�49X�ixս���aG����H�9�8Q콉7L�m�h�e`B��+�y�#�@���7L�D���q���m�h�����`���
��7L��E��y�#���-���w��Q콝�-��C���񪽑hs���w��{������ Ž�^5���T��l���;d����BlB��Bh�B�A��A�i�B��BT�B	Bx�By;B�3BMSBJ�B0�7A���B	j�B$�B1BN`B�B��B�KBܷB((4A���B#�B�'A���B
)B�A���B�B&��B%�B"O�B#)B�rB
�B8BJB�*B+SB �B�-B~�B,tBBޤB
�BB�B�xB��Bp*BA�B
�@BEB,��BM"B �B/�B*�AB�.BYMBőBC�B&KB%�gB��BeyB7OB@�B�B?�B?�A��A�O�B?�Be�B2�B�6BA�B� B ��BB�B0@�A���B	�B�B B?�B��B��B<B��B(;pA��|B@QB6�A��xB	��B�A�t�B��B&��B?`B!��B#|mB =�B	�KB��B��B�B+@B ��B�.B��B>ZB?�Bt�B	�B��B<-B�B@�B+qB
��B?�B,�kBAB��BU�B*ãB��B\B� B?jB&8�B%��B�yBėB?�@�$A��A9��AD�A׵�A�DAx�DB
1�A�W�A��AǍHAe|�A���A��4AX��B�A�]AK
~A�wA��A�?�A�CR>��A�+�@t'A���A��zA�%�A��A�S�A!()A�q�A��A���A�$R@BR�A�n�A��A[��A��A��AznAr�QA�h�A��A^��A�F�A���Ax�A���A%H=�h�>��0@�p?�`@A��iA��@��AU�?�A�6@�S�A�EMA��mA/�\BO�A%
/A4�0A���C���C�Q@�ԕA��A8�AD^�A�y�A��Ay�B
A�A�u�A�:�A���Ad�XB :�A��AX�B=SA���AKzA�Y�A���A��_A��^>��1A��?@s�A��A��lA�{fA��A� FA �CA���A�|�A���A��@9edA�|:A�{�AZ��AϬ�A�yAx��Aq�A��2A�MA]w�A��A��5Ay1�A�� A�>'ܴ>M�@�q?��WĀ�A�uh@��PAV��?��AAҏ_@�J�A�]A�y�A0��B A#�'A5� A�G"C���C��         
            &            	            '      
         -                                    	         ,                                             9   8   !      %   	         !      	   '            	                  
                     +                  '      -         /      )                           %               /      #                              #            )   -                     -         3                                                   #                  '               %                                 #               /                                    #            %   '                     )         3                              N�vN��?N���ON�NT�N�	O�H2N��LN �iN��@NE�$O(��P�N��]O}��O��cOP�EO��N<�O�DN,NYN
6WOM�rN���N�WN�~:M�N�w�O�5ZOqR\N���N��O
�zP+
O+sO�a]M��:N�TaN�
CO���N��vO/Oy�O�N�{O��
N���O{]�O��P�P #O~
O4�O��UN���N�=pO9��O�{RN��N2�PC��NIN&N���O8�lN7�KN�X7O�uNF�N��N���N(��  r  (  Z  �  U  ~  u  .  �      �    �  �  '  �  �  �  �  �  �  @  �  �  =  �  9  m  d  }  �  �     �  5  �  �  �  �  (  �  �  b  j  �  �    4    `  d  �  !  �  X  �  �  �  a  e  �  ?  �  7  ]  9  K  �  �   =o<�<�C�<u;�`B;ě��#�
%   ��o�ě����
�ě��ě��ě���1�t��o�49X�t���/�e`B�e`B��t�����������ͼ���������h��/�������o�H�9�t��t���P���''''49X�8Q�8Q�8Q�<j�8Q�aG��L�ͽL�ͽL�ͽ]/�Y��m�h�u�u�y�#�y�#��������C���hs��t��������-����ě��Ƨ��`B������������������������������������������������������������������

��������gmvz}���zmgagggggggg,/3;?;84/**,,,,,,,,,5:BN[glpqpmgNB(#/<?HLKH</'# ����������������������������������������7BOOX[\[OHB777777777��������������������w���������������~vuw��������������������!'2COS\`c_YIC6(HKTaju�����zm_TMHFHMU[gt�������tg[NIFLMsy�������������yyros�������������������������

�����������������������������������������������������269BCO[\htqhh[OHB622abnr{|{vneb[aaaaaaaa`aemzz|{|zzpmga__]``����������������������������������������BTm���������zmaHE<<B`g�����������tpgc\[`Y[fhituvutkh][VUYYYY*/;ACFC?;/+'$#******#$/;<=<::64/#�
#0<Un{���{U<#
���������������������������� �������������������������������������������������cgit�������tigcccccc���
#,/0/,
�����#/8;:3/#)36ABKMGB61)�������������������������������������������������������������������	�������:BDN[e\[NNB7::::::::p{���������������vtpdnz�����������zungddLTft�����������tgUKL��� '(&#�������1<HUahnrvxzrnaUC<:11NX[hn{�{|wtrh`[UOLNx{���������������zxx��������������������st������������tissss������������������������������������;<DHU__YVURHH<9669;;������������������������);������������������������������|ttppt�����������8BMNS^ilmjg[NEBAB@>8���	����������������������������!#&0<?IMPPJIF><0.#!!-038<@A<60.))+------��������������������#/<HLOKH<5/(#:<GHIRTLHE<;::::::::��u�s�t����������������������������/�&�#���
��
���#�%�/�<�A�H�L�H�<�/�4�*�(�����(�4�A�B�M�Z�\�Z�T�M�A�4�4�s�j�f�Z�^�]�a�f�s�������������������s�)�!�)�/�6�B�B�F�F�K�B�6�)�)�)�)�)�)�)�)��������������������������������������ؿѿĿ������������������Ŀѿݿ�������ݿ��$���"�$�*�0�=�A�F�F�=�5�0�$�$�$�$�$�$������)�)�+�)�����������������(�5�5�<�5�(����������a�Z�a�a�n�y�z�Â�z�u�n�a�a�a�a�a�a�a�a�;�2�.�!�"�&�,�.�;�G�T�]�`�m�v�m�`�T�G�;�������������5�C�O�\�r�v�h�Y�O�C�6��g�]�Z�N�A�7�7�A�N�Z�]�g�s��������s�g�g�����׾˾ɾʾҾ׾������,�.�%��	������Ƹƶ����������������$�$�����������������������������������	�����侌�s�f�Z�N�]�s�������ʾ׾������׾������g�b�b�e�g�n�s�}�|�s�g�g�g�g�g�g�g�g�g�g�������������������������������
��������"���"�/�;�<�;�;�/�"�"�"�"�"�"�"�"�"�"�������������	��	����������������������Ϲ������������ùϹܹ����������ܹ��
��������������������������
���
�
�!�!��!�+�-�3�:�?�:�8�-�!�!�!�!�!�!�!�!���������������	��"�#�/�7�/�*�"��	��������������������������������������������¦­²º¹³²¦��������������5�?�H�O�P�N�5��������������������)�,�*�!�������꽒��������������������������������������s�o�l�s���������������������s�s�s�s�s�s�N�L�B�7�5�+�5�7�B�G�N�[�g�t�{�~�t�g�[�N�������x�c�`�`�]�V�U�Z�s�����������������(�������� �(�5�A�N�Q�R�N�L�A�5�(�������ĺɺֺ�������������ֺ�������������������������� ����������������àßÔÓËÓàìñùý��ùìàààààà�	������������	��� ����	�	�	�	�	�	��ùìâàÖÖàìù��������������������H�A�@�H�U�a�n�w�z�n�a�U�H�H�H�H�H�H�H�H�Ŀ����������������Ŀѿڿ�����ݿѿĿy�v�n�m�w�������������ɿпͿĿ��������y�"�������"�(�/�;�F�H�O�N�H�;�/�"�"������������� �������������������������.�"��	����׾�	��.�;�T�W�c�`�T�;�.����������������������������������������ŹŠŘŇŀŇŔśŠŭŹ����������������Ź���������������Ŀƿѿֿۿ׿տӿѿͿĿ�����¿¦¦¿���������
���
���ؽ�����������!�:�S�`�m�y�����`�T�8�!���������������������Ϲܹ���ݹù��������ù��������ùϹܹ�����������ܹչϹú����~�t�Y�P�Y�a�e�~���������κź�������������$�'�3�@�I�@�?�3�'�"�����ììâàÛØàìù������������ùìììì����ĿļĹ���������
������
�������̼��������������������
� ��㼽�������׾־ʾľľʾ׾�����	�	��	�����׾׺3�1�3�=�@�L�Y�Z�Y�N�L�@�3�3�3�3�3�3�3�3ùÛÐËÝë�����)�B�J�>�2�.� �����ù�ܻջлû������ûлܻ���������ܻܻܻ�����������������������������������������ā�{āĂčĚĦĳĿ������������ĿĳĚčā�������������������������������������������������$�%�$��������彫�������������������Ľɽнݽ߽�ݽǽ�����������(�0�4�8�4�(������������������������������������������������ECE?EAE=E=ECEPE\EbEcE\EVEPEJECECECECECECD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� ; _ I 7 Q y : V K - ] 5 + x I 6 A b V @ ? Z M � 7 ^ { % ^ ( * G Q S - a � ` J ; : , ! D . d j " n > 0 V @ Y n L M l [ b F � i c F D / d B x 3      >  �  �  �  l  %  �    �  �  x  O    "  w  �     f    B  9  �  (  8    G  	  p  �  �  �  :  Z  =  z  w  �  �  W  �  O  �  #  *  �  �  �  �  S  �  1  o  E  �  �  �  g    R  �  �  �  �  D  0  R  �  �  �  D  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  >[  r  i  `  W  Q  K  F  D  A  A  B  B  ;  4  ,  #        �        %        �  �  �  �  �  �  �  �  g  @       �  Z  O  C  ;  3  ,  %        �  �  �  �  �  o  R  4  	  �  �  �  �  �  �  �  �  �  y  ]  4    �  �    �  �  �  �  e  U  U  U  T  R  O  I  @  7  !    �  �  �  �    a  B  $    ~      �  �  �  �  �  �  �  �  ~  z  u  q  l  h  d  _  [    2  O  d  m  t  t  k  a  V  H  3    �  �  U  �  ~  �   �  .      �  �  �  �  �  �  ~  d  G  (    �  �  ]    �  �  �  �  �  �  �  �  �  �  �  �  l  T  <  $    �  �  �  �  �  �  �  �                             
                �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  `  p  �              �  �  �  �  �  �  �  �  b  B  -    K  `  �  �  �  �  �  �  �  �  }  m  ^  P  B  5  '      �  �  �  y  �  �  �  �  �  �  �  �  �  �  �  �  �  �  A  �  z  �   ]    '  #      �  �  �  �  l  K  +    �  �  �  0  �  2   �  �  �  �  �  �  t  ^  F  ,      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  b  P  5    �  �  o  )   �  �  �  �  �  �  �  �  �  �  �  {  q  f  \  R  G  =  3  (    �    N  }  �  �  �  �  �  �  �  Z  )  �  �  8  �  �  "  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  @  ;  0       �  �  �  i  <      �  �  �  C  �  F  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  {  z  y  w  v  �  �  �  �  �  �  �  �  �  �  m  +  �  �  1  �  k    �  *  =  -      �  �  �  �  �  m  M  ,    �  �  �  �  t  a  M  �  �  �  �  �  �  �  �  �  �  �  �  z  n  c  X  M  A  6  +  9  4  ,  !    �  �  �  �  c  -  �  �  �  B  �  �    �    b  j  m  i  `  T  D  4    �  �  �  P    �  �  �  Z     �  d  ^  Z  X  V  R  R  N  H  ?  4  #    �  �  �  ~  J    �  }  {  y  x  r  l  f  `  X  Q  I  @  7  ,      �  �  ;   �  �  �  �  �  �  �  �  }  m  ]  M  <  ,         	  
      �  �  �  a  8    �  �  }  L    �  �  �  �  t  #  �  q         �  �  �  �  �  k  E    �  �  �  U  2  /  $  �  �    �  �  �  �  �  �  �  �  �  �  �  �  j  7  �  �  1  �  =  y    ,  5  4  .  "    �  �  �  �  P    �  �  i  -  P  w  Q  �  �  �  �  �  �  �  �  �  �  �  x  i  [  M  ?  0  "      �  �  �  �  �  �  �    m  X  B  +    �  �  �  �  �  D    �  �  �  �  �  �  �  �  �  �  �  o  M  ,  	  �  �  �  n  D  o  �  �  �  y  a  C    �  �  �  >  �  �  _    �  �  B  �    "  (  %  !                  �  �  �  �  q  L  �  �  �  �  �  j  Y  E  1      �  �  �  �  t  =  �  �  i    �  �  �  �  �  }  n  Z  C  +    �  �  �  O    �  �  3  �  b  L  5    �  �  �  �  �  {  S  &  �  �  u  5  �  �  �  O  j  g  c  `  ]  Z  W  T  Q  N  F  :  .  "    
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  U  /  �  �  `  	  �  d  �  �  �  �  �  �  y  o  f  ^  V  M  E  =  6  1  ,  '  "          	    �  �  �  �  �  �  |  c  I  -  
  �  �  {  7   �  4    �  �  �  �  �  `  ;    �  �  �  k  9    �  �  D  �  �  �        �  �  �  �  �  ~  r  Z  -  �  �    ~  �  >    `  [  Q  F  8  "  �  �  ~  2  �  d  �  �  �  Z    "  �  =  b  Q  *  �  �  �  E  �  �  O  �  �  ~  S    �  L  �  _  i  }  �  v  T  2        �  �  w  c  M  7      �  �  ;  !        �  �  �  �  i  9    �  �  V  �  c  �  �  9  �  �  �  �  �  �    w  p  X  >  "    �  �  �  o  D    �  �  W  W  L  >  *    �  �  �  �  [  +  �  �  �  R    �  I  �  �  �  �  �  �  �  �  �  �  �  y  m  `  L  >  i  �  m    �  �  �  �  �  }  W  3    �  �  �  e  3    �  �  $  �  �  t  �  �  �  �  �  �  �  n  F    �  �  v  ?  �  �  #  �    t  N  V  ^  Y  L  >  0  !  >  a  m  a  V  J  =  0  #      �  e  8  @    �  �  C  �  �  �  {  :  �  �  l  )  �  �  �   �  �  �  �  �  �  �  u  \  >       �  �  �  �  y  V  ,     �  ?    �  �  �  �  �  }  d  H  )    �  {    �  s  0  S  �  �  �  �  ~  c  v  X  3    �  �  �  Q    �  �  �  @  �  �  3  5  6  /  #      �  �  �  �  �  �  `  @    �  �  �  �  ]  I  2    �  �  �  �  m  P  8    �  �  �  u  P  2  '  #  3  7  5  '    �  �  �  �  S  #  �  �  �  Z  *  �  �  l  6  K  A  7  -  #        �  �  �  �  �  �  �  �  �  �    t  �  �  {  c  A    �  �  p  7  �  �  �  C    �  �  c      �  �  �  �  �  �  �  m  X  ?    �  �  �  C    �  �  P         �  �  �  �  �  d  F  "  �  �  �  P    �  �  U    �
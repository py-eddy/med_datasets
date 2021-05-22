CDF       
      obs    G   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?���vȴ:       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�0%   max       P�L�       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��E�   max       <e`B       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?���R   max       @F333333       !    effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�=p��
    max       @v��Q�       ,   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0�        max       @Q�           �  70   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�`            7�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �I�   max       <D��       8�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�P   max       B0-�       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B0?�       ;   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?��.   max       C���       <0   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?���   max       C���       =L   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          5       >h   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;       ?�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          '       @�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�0%   max       P*]�       A�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��X�e   max       ?������       B�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       <e`B       C�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�z�G�   max       @F*=p��
       E   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��
=p�    max       @v��Q�       P(   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @Q�           �  [@   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�/�           [�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @�   max         @�       \�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�ݗ�+j�   max       ?������     0  ^      
                                             *                  4                1   -                     
         	                           !                     "         	                                       1M��pN�
N��N�4�NXґN�w�O��OT�$N��N��O.I�O�tKO�;P:�bNDK{O&C�P4%�N�vOA�N"w�O��O_{�P�L�O��\O%gOPrO5VONP'�O�JO�]NN��O�� N�2SOw��N�P�OA�Nc/�N���O���O��N]?5O1�N�0�O&kcO ��N���O�;IO��;NNS�N^ڦOm�NJ3�NjYsOZ3XO�b�N��)N�I�N�=�N�XaO55O8��N�,M�0%O1_N�`N�NfxUO΋N���Ofq<e`B<#�
;ě�;ě�;ě���o��o��o�o�D�����
��`B�o�o�49X�e`B��o��o��C���C���C���t����㼛�㼛�㼣�
���
���
��1��9X���ͼ�����/��/��`B��h�������o�t��t��t���P��w��w�#�
�'0 Ž0 Ž@��H�9�P�`�]/�]/�aG��e`B�e`B�ixսq���y�#��������O߽��P�������
���T���T��E���E�#+/0/$#"������


���������NNOR[gjkmgb[SNNNNNNN��
#(/7<</#
������������������������������������������������
"%"#.#
�����16@O[ht~tkjlh[OKE711"#$(/5<DC?</-#""""""�������


��������rt������������xtolmr.<HUan������zznU<,(.��!"������������������������{|�?BNQVXNCB>??????????:?FHU_ajnnjjcaUPH@;:Wm����������rma\ZUUW��������������������')-5BN[egc][TNB5*)''���

���������*6COTWXXVOC6,fgty�������tgb^^\]^f��Bt������gN5�����������# 
��������EIUbln{�|smdZULIC?EZaimrz}�����zmiaZUVZ�����
#.'#
�����������
 
��������������������~��#/<HUUY[WUH<5//)####)5BLX]]VMHB=5)��������������������jpz�����������zmiggj��������������������pt���������������{zp�������������������������������������������� 


�����������Y[^hrt���tmhc[YWYYYY{�������������}{wwy{�����
����?BCO[hkhd[OB????????38;BIUVRQSSOI=<0,')3������������������������������������������������������������@BIO[hlty|th[[OLFB@@)5[tyzuth[OH/)�������� ��������	!$"								����

����������35BKNT[[[[VNB:5.+-33[[hltwvth[ZY[[[[[[[[yz�����zvuyyyyyyyyyy�������������������������3;7)��������	�����������*0<IU]]USJI<70+-01'*
#(/:/#
�����������������������������������������������  ��������	����������pt���z���tsqpppppppp5=BN[ehge^[RNDB53+.5w�����������}wwwwwww��������������������"#&*+*((#!,/1=HUafiiaUMHC<:/-,������������������������������������������������ �*�+�*�!������������ùìáàåìùù�������������������žf�e�Z�M�C�F�M�Z�f�o�s�w�s�g�f�f�f�f�f�f�����#�'�)�2�6�A�C�E�E�B�A�:�6�)�!��!����!�-�6�:�F�G�F�:�.�-�!�!�!�!�!�!��üù÷ù�������������������������������s�g�Z�P�H�D�=�A�N�Z�\�g�i�c�f�o�s�v�t�s�U�L�H�L�Y�_�X�a�f�zÓàåçÝÇ�z�n�a�U�����ݽԽݽ޽���������������������s�p�f�\�Z�P�N�R�Z�f�g�s�z�v�s�r�s�t�s�s�m�h�a�a�l�m�y�����������������������y�m�4�(�������(�A�Z�f�h�f�]�^�U�M�A�4��������)�+�5�>�B�M�B�9�5�)��������s�Z�T�C�B�M������׾�������㾾���}�}�����������������������������������������������(�5�A�D�C�A�>�5�(����ٿտۿ����(�.�<�N���������N�5���������������������ĽĽĽĽ��������������������������������������%�"������v�t�q�t��	������޾�����"�.�H�S�P�G�;�0�"��ĿĿ¿Ŀɿݿ���
����������ݿѿľ��ʾ����˾ƾ׾��	��$�&�.�%�.�-�"����׾ʾľǾѾ������"�3�0�2�+��	���׼����������������������ʼּ����ּʼ��t�k�t�|āĈčĚĦıĳĵĵĵĳĪĚčā�t�Ŀ��������������ĿѿԿݿ������ݿѿ�EEED�D�D�D�D�D�EEE*ECEPEREOECE7E*E�#�
���#�0�<�nŇŔŠūŢŉ�{�n�b�U�<�#�
� �� �
�
��#�/�9�7�6�/�$�#��
�
�
�
�5�����(�5�A�N�Z�g�p�s�|�}�u�Z�N�A�5ùîìååìòù������������ùùùùùùƳƧƙƕƎƎƚƧƳ�������������������Ƴ�*�)���
������*�1�6�<�6�+�*�*�*�*ŹŰůŭšŜŢŭŹ��������������������Ź�ѿ̿˿ѿۿݿ������������ݿѿѿѿѼM�J�F�=�@�M�Y�r���������������r�f�Y�M�������������#�'�4�;�4�'��������Y�P�Y�^�e�g�l�r�x�|�~���~�y�r�e�Y�Y�Y�Y��ѿĿ��������ĿͿؿݿ�� ��(�/�(������������������������������� �	���	���������������������������������������������������s�k�m�s���������������������������t�l�r�q�t�t�t�t�t�t�t�t�t�����������������(�)�.�6�;�6�)����������������������������������޻l�`�l�n�q�v�x�~�����������������x�q�l�l�s�p�\�g�y�����������������������������s�������������������ʼּ�߼ּмʼ����������������!�&�$�!�������������������ɺĺ��������ɺϺֺ����ֺɺɺɺɺɺ��������*�*�6�8�C�G�F�C�?�6�*����������'�)�,�+�'�����������ߺֺ�������������������~�y�r�h�i�r�~���������ĺɺԺɺ��������~�����y�[�`�l�m�y�������Ľѽ޽��۽н����z�o�n�i�h�n�zÇÌÒÍÇ�z�z�z�z�z�z�z�z�-�*�*�)�*�-�:�D�F�S�U�]�_�a�_�S�N�F�:�-DbD[DXDbDjDoDxD{D�D�D�D~D{DoDbDbDbDbDbDb�����������������������ûûɻû»���������������������������������
����������!����������������!�$�+�-�,�%�!���������������ʼּ�ּּʼ¼�����������ĦĠĥĦĨĳľĿ����ĿĳĦĦĦĦĦĦĦĦ�a�U�O�H�E�H�U�W�a�n�q�z�~ÇÉÈÇ�z�n�a�<�1�0�;�<�H�U�a�h�e�a�V�U�H�<�<�<�<�<�<�_�[�Y�_�l�x�}���x�l�_�_�_�_�_�_�_�_�_�_E�E�E�E�E�E�E�E�E�E�E�FE�E�E�E�E�E�E�E�FFFFF$F1FAFJFVFcFpFoFcFZFJF9F1F!FF�������������ûлܻ��ܻлû���������������������������'�+�0�-�'����� { H 6 w < 4 { ] * S " W 0 M U ! f 7 3 s " 6 Z N q T ` Z Q < ; B @ B $ g R o g z ; ^ o 8 M 9 ~ L X I 7 , : = N > - K R @ 5 e T q Q ; Y w � P (    _  �  �  9  o  �  �  �  �  �  u  �  M  �  \  `  �  �  �  f  ^  �  )  .  �  _  �      @  g  �  �  �  �  �  �  �  �  �  6  �  �  �  }  o  "  C  C  n  m  0  ^  h  �  �  �    �    H  �  �    p  �  *  �  �  �  4<D��%   :�o��o%   �49X��`B�����t��o����C���h��㼋C����m�h��j�0 ż��
�8Q�t���hs�L�ͽ\)�\)������O߽���\)�49X�,1�<j�o����w�T���C��#�
�]/�@��''0 Žm�h�]/�aG����P�}�L�ͽ]/�y�#�}󶽋C���-��{��o����������P��t����T�� Ž�hs��Q�����{�ě��Ƨ����I�B~�B�B��B[�B,ǓBC`BNB��Bh�B��B�B��BnaB �3B�B��B E�B!D�B؁B��B0-�B	�zB��B��B'F�A�PBs�B��B�{B�XBZEBm�B u�Bq BX'B�@B"U B#s�B�4B*��B�B�B&V�B|B� B"�BkB��B,�3B.��B#��B�B�|B��B�_B'�B��B&i�B^oB!ϊB��BZ�B2gB	��B�B
�BBB�mB@�B�B�4B�vB>�B��B��B,�B �'B�B��B��B?'BSB��B� B! �B@B��A��B!=�B��BWB0?�B	�BAB>�B'=zA��B�OB�|B�^B��B}�Br�B ?PBB�BK�B��B"?�B#�&B��B*Q�B-zB��B&@�B
�oB��B=�B��B��B,C�B.��B#��B��B��B�:B��B>�B�yB&�EB<_B"<�B��B��B?tB	ˀB2CB
�rB)(B�B�BE�B��A�]�A͢A?�MA��h@s[AϞ�A���A�ߤA-�A@M�AnK�A:CtA� )AH2�Ap�EA��"A�U	A$�Ba�A��oA\ݨA�OAW0AX`5@�f9A��+Az}�C�}A��A���A�0�A͂?BPA��\A�u�A~�|@�ҥ@Ê�?�M�A~:�A�6gA���A��3A�bA��cA�J�@�PEA���@���A	��@9�0A�a�?��.@LW@čA#�AȞ�@�C��2@��QA��NA
Y�@��A�p�A��Aĳk@�s�C�qFC���@�9@��A�pdȦ=A@�A׀i@tiAπ�A�|A�n�A-.�A@�=Am�A:�2A�{|AE�>Ar�PA��A�� A$�<BAA�j A[A��|AW�AV��@���A�eAzGC��A�T�A��A�O9A͋BDA���A��*A}G@��@�/�?�iA{XA���A���A�_�A�qKAԗ�A�S�@� �A��_@�A	N�@4�A���?���@LC�@�fA"��Aȥ�@{��C��a@�A+A�zA
�=@�
A��(AƄ�A�z�@���C�g�C���@��#@�	�                                                   +                  5                2   -                              
                           "                     "      	   
                                       1                                    !      3         1                  ;   %               -                                 )                        %                        %                                                                                       #         !                  '                  !                                                         #                        !                                             M��pN�
N��N�vNXґN�w�N���NZ�N��Nk��N�QN�AaO�;O�� NDK{N��Oʻ�N)��O�KN"w�O`ФO>�P*]�O��N���N��\O5VO;pO�bkO�JOGdN���O�� N�2SN�00N�P�OA�N)?N���Or:�N�IeN]?5O1�N�0�O&kcN��MN���O��'On(LNNS�N^ڦOm�NљNjYsO@�O��dN��)N�I�N�=�Nɑ�Nh�O8��N�,M�0%O1_N�`N�NfxUO΋N���N�b�      �  �  �  <  L  �    ~  �  V  �  �  �  �    =  �  �  �  1  B      �  1  S  4    a  7  7  N  1  �  �  �  �    �    Z  x  �  �    T  �  U  w  �  �  "  �  �  �  �  w  y  �  �  W  �  &  %  �  �  �  �  �<e`B<#�
;ě�;��
;ě���o�o�D���o��o�D�����
�o��o�49X��C���h��t���9X��C���j���
�o������j��1���
��9X�+��9X�+��`B��/��/�o��h��������P��w�t��t���P��w�49X�,1�49X�49X�0 Ž@��H�9�Y��]/�y�#�ixսe`B�e`B�ixսu��o��������O߽��P�������
���T���T��E����#+/0/$#"������


���������NNOR[gjkmgb[SNNNNNNN�
#&/37/#
��������������������������������������������������
 
������@BEOY[b[OB?9@@@@@@@@"#$(/5<DC?</-#""""""�����

 ���������������������}uv������AHTUajnwuniaUHE=AAAA��!"���������������������������?BNQVXNCB>??????????BHOUVaglgfaUHD>=BBBB^coz����������zma]\^��������������������15BN[_][XONB5/**1111���

��������� *6CMQQQOLFC6*agt���������tgfa`^`a�5Ngt����gN5&����������

���������CIIUbkhb`UIGCCCCCCCC[agmpz}��zmka`[VW[[�����
#.'#
��������

���������������������������#/<HUUY[WUH<5//)####!)25;BINPQNLDB5,)$ !��������������������jpz�����������zmiggj������������������������������������������������������������������������������������ ��������������Y[^hrt���tmhc[YWYYYY���������������~�������������?BCO[hkhd[OB????????38;BIUVRQSSOI=<0,')3������������������������������������������������������������IO[hitwzth][ONIIIIII)-6B[jtxtrh[O2)!�������  ����������	!$"								����

����������35BKNT[[[[VNB:5.+-33Z[`hjtuuth[[ZZZZZZZZyz�����zvuyyyyyyyyyy�������������������������)294'��������	�����������*0<IU]]USJI<70+-01'*
#(/:/#
�����������������������������������������������  ��������	����������pt���z���tsqpppppppp5=BN[ehge^[RNDB53+.5w�����������}wwwwwww��������������������"#&*+*((#!,/1=HUafiiaUMHC<:/-,������������������������������������������������ �*�+�*�!������������ùìáàåìùù�������������������žf�e�Z�M�C�F�M�Z�f�o�s�w�s�g�f�f�f�f�f�f����$�(�)�4�6�B�D�D�B�@�8�6�)�����!����!�-�6�:�F�G�F�:�.�-�!�!�!�!�!�!��üù÷ù�������������������������������g�_�Z�R�N�J�F�@�A�N�Z�`�d�g�m�p�s�u�s�g�U�T�U�X�a�e�n�x�z�q�n�a�U�U�U�U�U�U�U�U�����ݽԽݽ޽���������������������f�f�Z�S�Q�X�Z�]�f�s�w�s�s�f�f�f�f�f�f�f�m�l�k�m�y�|���������������y�m�m�m�m�m�m�4�-�(�(�'�(�)�4�A�H�M�Q�T�O�M�A�4�4�4�4��������)�+�5�>�B�M�B�9�5�)����s�f�[�T�O�Z�f��������žʾپѾʾ������s���}�}�������������������������������������������(�5�=�?�:�5�(�������������������(�2�A�M�Y�I�5������������������������������������������������������������� ��������������v�t�q�t�	���������	��"�.�;�>�A�=�6�.�"��	�ѿȿĿͿѿݿ��������������ݿѾ׾ƾƾ����վ۾پ������%�"���	���׾׾Ѿʾ̾־����	��$�&�&�"��	�����׼ʼ����������ʼӼּ�ڼּʼʼʼʼʼʼʼ��t�m�t�}āĉčĚĦĳĴĴĳĨĦĚčā�t�t�Ŀ��������������ĿѿԿݿ������ݿѿ�D�D�D�D�EEEE*E7ECEPEQENEBE7E*EEED��I�<�0�#���#�0�<�U�n�{ŇōŇ�{�n�b�U�I�
� �� �
�
��#�/�9�7�6�/�$�#��
�
�
�
�5�)�����(�2�5�A�H�N�Z�_�`�Z�X�N�A�5ùòìæçì÷ùý��������ÿùùùùùùƳƧƙƕƎƎƚƧƳ�������������������Ƴ�*�)���
������*�1�6�<�6�+�*�*�*�*��ŻŹŲŭũŭŹ�����������������������ƿѿ̿˿ѿۿݿ������������ݿѿѿѿѼM�J�F�=�@�M�Y�r���������������r�f�Y�M���������'�0�'�����������Y�P�Y�^�e�g�l�r�x�|�~���~�y�r�e�Y�Y�Y�Y�ѿĿ����������ÿſݿ��������������������������������������������������������������������������������������������������s�k�m�s���������������������������t�l�r�q�t�t�t�t�t�t�t�t�t�����������������(�)�.�6�;�6�)�������������������������������������ѻx�q�t�x�x�������������������x�x�x�x�x�x���s�c�g�i�p�{�������������������������������������������ʼּݼּϼʼļ������������������!�&�$�!�������������������ɺĺ��������ɺϺֺ����ֺɺɺɺɺɺ��������*�*�6�8�C�G�F�C�?�6�*�����������'�'�+�'�'����������ߺֺ������������������⺋���~�v�r�q�q�~���������������������������������s�q�y���������Ľн۽�޽ؽнĽ��z�o�n�i�h�n�zÇÌÒÍÇ�z�z�z�z�z�z�z�z�-�*�*�)�*�-�:�D�F�S�U�]�_�a�_�S�N�F�:�-DbD[DXDbDjDoDxD{D�D�D�D~D{DoDbDbDbDbDbDb�����������������������»ûȻû������������������������������������������������!����������������!�$�+�-�,�%�!���������������ʼּ�ּּʼ¼�����������ĦĠĥĦĨĳľĿ����ĿĳĦĦĦĦĦĦĦĦ�a�U�O�H�E�H�U�W�a�n�q�z�~ÇÉÈÇ�z�n�a�<�1�0�;�<�H�U�a�h�e�a�V�U�H�<�<�<�<�<�<�_�[�Y�_�l�x�}���x�l�_�_�_�_�_�_�_�_�_�_E�E�E�E�E�E�E�E�E�E�E�FE�E�E�E�E�E�E�E�FFFFF$F1FAFJFVFcFpFoFcFZFJF9F1F!FF�������������ûлܻ��ܻлû�������������������������'�*�/�,�'����������� { H 6 s < 4 ~ R * : # A 0 6 U  C C * s   7 S G < S ` X 4 < # ? @ B 0 g R ` g f 4 ^ o 8 M 4 l H Q I 7 , 7 = > ) - K R C > e T q Q ; Y w � P     _  �  �    o  �  F    �  �  �    M  
  \    �  D  %  f  �  �  .  h  �  /  �  �  �  @  E  �  �  �  �  �  �  V  �  t  �  �  �  �  }  �  �  �    n  m  0  7  h  e  y  �    �    l  �  �    p  �  *  �  �  �  
  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    �  �  �  �  �  �  �  �  �  i  M  .    �  �  �  G  =  )  �  �  �  �  �  �  �  �  �  �  �  �  y  e  P  <  *       �  �  �  �  �  �  �  �  �  `  .        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    m  \  J  8  '       �  <  9  6  .  %        �  �  �  �  �  f  \  _  b  d  �  b  F  H  J  L  >  -      �  �  �  �  �  �  w  c  L  (     �    �  �    G  y  �  d  �  �  �  x  g  L  &  �  �  �    p                !  %  &  '  '           �  �  �  �  \  e  n  w  ~  |  {  z  x  u  r  o  g  \  P  E     �   �   a  }  �  �  �  �  �  �  �  �  �  �  �  �  |  ^  ?    �  �  �  �  �  �  �  �       1  H  U  O  <     �  �  �  :  �  m  �  �  �  �  �  �  j  N  /    �  �  v  2  �  �  I  �  L  �  �  #  4  G  w  �  �  �  �  t  b  E     �  �  �  o  ;  �  �  f  �  �  �  z  l  ^  O  @  1  "       �  �  �  �  k    �  _  �  �  �  �  �  �  �  �  �  �  �  �  ^  2  �  �  �  W  ;  U  �  �  �  �  �  �      �  �  �    L    �  r  �  #  c  �  0  3  6  8  :  <  <  :  8  2  +  $        �  �  �  ~  O  u  �  �  �  �  �  �  �  i  K  '  �  �  �  J  �  [  �  �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �         �  �  �  �  �  �  �  �  �  �  �  y  U  &  �  �  `    �  E  *  /  1  *    
  �  �  �  �  �  k  A    �  �  �  n    �  �  �    ,  ;  =  #    �  �  v  9    �  �  X    �  �   �  �  �  �  �      �  �  �  �  �  _  ,  �  �  [    �      �  �  �  �  �  �  �  �  �  �  �  ~  _  1  �  �  8  �  �  !  �  �  �  s  P  +    �  �  �  �  p  W  6    �  W    �  l  1      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  
o  N  2    
�  
�  
:  	�  	�  	`  	I  	  �  �  6  �  �  �  �  �  t  �  �    $  3  /    �  �  �  �  g  $  �  �    P  �  f    �  �  �  �  �  �  �  �  �  �  r  W  <  $    �  �  �  U  4  5  7  6  ;  B  P  ]  a  ^  R  <  !    �  �  w  +  �  h  �  '  6  5  0  '    	  �  �  �  �  t  3  �  �  ]    �  A  7  2  +  #          
    �  �  �  �  �  u  1  �  i   �  N  J  F  B  =  6  0  )  !        �  �  �  �  �  �  �  }  $  %  &  '  &      *  .  &         %  $      
  �  �  �  w  l  ]  O  @  0      �  �  �  �  c  -  �  �  z  :   �  �  �  h  K  +    �  �  �  �  �  u  ;  �  �  J  �  �  T  '  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  [  K  3    �  �  �  ]  .    �                �  �  �  �  j  >    �  x  8  �  }  A  r  �  �  �  �  �  �  �  �  �  �  �  �  �  �  d  6  �  �  l    �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  f  Q  <  '  Z  P  G  =  5  5  4  4  3  1  .  +  "       �   �   �   �   p  x  j  ]  P  L  K  J  B  8  .  !      �  �  �  �  �  �  �  �  �  o  U  6    �  �  �  l  <    �  �  :  �  G  �  6  �  �  �  �  �  �  �  �  �  �  �  �  p  M     �  �  d    �  4  �  �  
  
  �  �  �  ~  i  a  =  9    �  �  R  �  �    �  E  R  R  H  5      �  �  �  v  8  �  �  �  p    ^  �  �  t  �  k  P  /    �  �  �  b  0  �  �  t  <  �  �  �  5   �  U  A  ,      �  �  �  �  �  �  �  j  S  8    �  �  �  r  w  j  ]  O  @  1  #        �  �  �  �  �  ~  k  Z  K  <  �  �  �  �  �  �  �  w  ]  A  #    �  �  �  p  B    �  s  �  �  �  �  �  �  �  �  �  d  7    �  �  J    �  t  (  �  "    �  �  �  �  f  B    �  �  �  d    �    (  �  u    U  e  }  �  �  �  �  ~  d  ;    �  o    �  -  �    ~    �  �  �  �  �  �  �  g  H  #  �  �  �  a  ,  �  �  x  �   �  �  �  �  }  k  X  D  0      �  �  �  �  _  ?      �  �  �  �  �  �  �  �  �  �  �  x  S  /    �  �  �  S     �   t  w  i  c  _  I  -    �  �  �  j  8  �  �  �    �  �  `  �  w  x  y  u  n  d  q  w  t  g  [  M  >  /       �  �  �  �  g  k  {  �  �  �  �  �  �  �  n  T  7    �  �  �  u  H    �  �  q  R  /    �  �  �  �  �  �  �  �  o  @  �  �  g    W  N  C  <  4  *      �  �  }  =  �  �  4  �  d  �  �  �  �  �  �                     �  �  �  �  �  �  �  �  &    	  �  �  �  �  �  m  M  +    �  �  f  (  �  �  _  �  %  #                %  (    �  �  s    �  c    �  8  �  �  �  �  �  �  �  �  �  �  �  �  �  �    q  @    �  �  �  �  ^    �  �  >  �  �  �  z  d  <    �  ~  6  �  �  \  �  R  
  �  �    �  �  �  �  a  7     �  V  �  �  &  �  J  �  �  �  �  �  �  �  �  o  Q  1    �  �  �  �  f  6    �  k  �  �  �  q  `  I  +    
�  
{  
  	�  �  F  �  �  5  �  �
CDF       
      obs    ?   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�j~��"�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P��R      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �o   max       =���      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>ٙ����   max       @F
=p��
     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�         max       @vq\(�     	�  *x   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @P            �  4P   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�3        max       @�+�          �  4�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ;�`B   max       >E��      �  5�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B.9�      �  6�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�~�   max       B.r�      �  7�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�j   max       C�f       �  8�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?��X   max       C�g�      �  9�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          o      �  :�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          G      �  ;�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          %      �  <�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       O�yV      �  =�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�&���   max       ?�%��1��      �  >�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���
   max       =�S�      �  ?�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�G�z�   max       @F
=p��
     	�  @�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?ᙙ���    max       @vq\(�     	�  Jx   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @P            �  TP   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�3        max       @�           �  T�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         =�   max         =�      �  U�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Fs����   max       ?�%��1��     �  V�                              	      b         
                              /   ,   "         &      (               	   
   -   /         $   !         #                  
               	   &         ,   oO��O�n�N�)�O�ܠN�mfO�UO]¶N��O���N5P�OgY�P��ROd�wM��N	vN�OyY	N.�OE��O,9\N�O�OK�N�J=N2��O�U:O-Oژ�OKuNWQ.O��xO'tO֏�N��DOQ]N+KGN
:�N第OgFO���O���NTO�?O�AN�F%O��O�lO(N�aN�j�O[A�N�A�Oj�6N�7�Nָ�NJ�O?b~OCiN�\O�1N\�PN�ԅO�N�O��3�o��`B�ě�:�o;D��;�o;�o;�o;��
;ě�;ě�;ě�;�`B<o<t�<#�
<49X<49X<49X<49X<D��<D��<D��<T��<T��<T��<e`B<�o<�o<�C�<�C�<���<���<���<�1<�9X<�9X<�9X<�9X<�j<���<�/<�`B<�<��<��=o=o=o=+=+=�P=�w='�=49X=T��=T��=�o=�C�=�C�=�t�=���=���>BN_dgt�������gdTKH>cbehkmz����������zmc����������������`[`ntz�����������}n`������������������������������� ��������������"'"%��dht����tohdddddddddd=<>CN[gtv��ytg[ZOB=OMB?BFOY[[_[[OOOOOOO)BNQUXYNB5+)"�����
<���za</
���������
#/<?5/)(#
����������������������� ������������������������������?:;@BMN[grtsqkg_[NB?VRZ[agnjg[VVVVVVVVVV��������������������������������������������%%�������,5BN[`glmg[NB@>=:75,B7BMO[ahtsrjh[OBBBBB�����������������������)6BKPV[E6)
�[Y[_eht���������tha["/6KagidN;/"5BN]Z[gg[NB52&##00330'#GILSchtv�������th[OGICCJO[]`ehkmomh^[ROI�������������������������������������+(().0<?FMQSSOIF<90+����������������������	����������)*557965){v�����������������{#<HO[dgUMJH</��������������������)5651)��������������������%"%/5N[ghhfc][NB50+%�}{������������������
)5BQXVMB4)��������*5850*�����#'/<=@BC@<4/##)6BOZVOB6)��������������������LNNIN[gmt���vg[VQNL����������HEECGHUahmz{{wonaXUH�����
# 
�����! #$0<@IKJID<50(#!!		
#&$#
								����������%")6BGOZ[`[VOB6-)%%���������������������������������������][amnvz~��~zmmheca]]rvoz��������������|r��������

����������)�6�8�)��������ùÿ����������������������������������������������������û˻лܻ����޻ܻлû����������������������������������������z�m�g�m�z�������(�0�4�:�7�4�(������
���$�(�(�(�(��������6�O�a�f�O�J�=�*�������������߿.�;�G�`�m�y�����������y�m�`�T�G�;�(�"�.�������������������������������������������)�5�F�I�E�8�5������������������l�l�n�y���������������z�y�l�l�l�l�l�l�lƎƒƚƞƚƣƟƚƎ�u�h�\�W�X�\�^�h�uƁƎ��(�/�A�K�I�L�+����޿������ѿ�����`�m�y�������������y�m�`�G�;�0�;�G�L�\�`�(�5�A�N�N�N�B�A�;�5�(�(�(�(�(�(�(�(�(�(ù������������ùõòùùùùùùùùùù������������������������������������������$�0�=�H�?�4�$���������������������������������������������������������������ûлܻ����������ܻлû�����������&�3�8�<�>�A�@�3�'���������� ��m�y�z�����������y�m�j�h�d�i�m�m�m�m�m�m�m�o�n�r�o�`�Z�G�;�.���*�.�;�G�T�`�d�m�<�H�M�U�U�_�a�i�a�U�H�<�;�5�5�;�<�<�<�<ƳƸƽƾƵƳƱƧƧƣƧƯƳƳƳƳƳƳƳƳ���������¼��Ƽ�������r�Y�M�H�O�Y�f�r���@�L�Q�Y�e�k�r�p�e�e�Y�L�@�;�3�1�-�.�3�@�"�/�T�a�g�p�r�m�T�I�H�;�/�����	��"Ŀ������������ĳĦġěęĘęĒĕĚĦĳĿ�ּ�����������޼ּԼּּּּּּּֽ�����4�A�<�4�/�(����ݽнĽ½н�����������ľ¾�������������s�g�s�t������ݿ����!�(�1�4�0�(������ؿͿɿʿѿݿT�`�c�m�y�{���y�m�m�`�T�O�K�N�S�T�T�T�T���Ľнݽ����ݽнĽ������������������������	�	���	���������������������¿½¿�����������������������t¦²µ²§¦�t�s�m�p�t�t�Z�f�w�����������������s�f�Z�V�R�N�W�Z��������� �����������ïô�������޾(�A�M�Z�g�l�t������s�Z�M�A�4�(����(�f�m�n�o�m�f�c�Z�V�U�Z�]�f�f�f�f�f�f�f�f�/�<�H�K�U�Q�H�D�?�<�/�(�#�!��!�#�*�/�/�������#�(����ݿѿĿ¿����ſѿݿ�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��#�<�b�{ŅŉŇŀ�{�n�b�U�N�I�=�6�!���#�m�y���������������y�l�`�S�M�G�I�L�S�Z�mD�D�D�EEEEE(E"EEED�D�D�D�D�D�D�D��m�z�~�������������z�v�s�u�s�m�j�m�m�m�m�����������������������������~�z�������������������������������s�g�Z�H�M�Z�s�|���/�;�H�H�T�V�U�T�H�;�/�"�"��"�*�/�/�/�/������������������������������������������)�5�B�C�N�O�N�M�B�5�)� ��������f�r���������������r�n�f�_�Y�V�Y�`�f�f�������ľǾ������������������������������:�F�S�_�g�l�w�x�x�l�_�S�F�9�-�!���-�:�:�F�S�U�U�T�S�N�F�C�:�5�-�(�#�!�-�4�:�:�����%�������������������'�3�@�O�Y�_�Y�L�@�'������������r�s�~����������������~�r�q�h�h�r�r�r�r������������������ŹŭťŬŭŹ���������߼����ּ߼������ּʼ���������������D{D�D�D�D�D�D�D�D�D�D�D�D�D{DzDtDsDuDyD{ 7 ' P x K T a U $ J + m B ^ W X . $ � > B B ^ } $  + ^ 5 * ^  2 N } } O R , = ^ M /  i \ 7 � a 8 8 5 4 * 8 9 B = F } p Z     �  1     �  �  ?  �  K  ?  Y  �  �  �  ,  B  $    +  '  �  �  �    �  �  j      n  �  X  �  �  u  �  �  #  �  �    �  >  t  	  �  �  Y  ?  �  �  �  �  �  �  `  �  <  ?  �  �  #  U  <49X<T��<49X<�t�<#�
<��<49X;�`B<���<e`B<���=��<ě�<D��<���<�1<�/<u<�<�j<�t�<��<���<�t�=u=e`B=D��=0 �<�1=aG�<�`B=q��<�9X=0 �<�j<���<��=o=�7L=�O�<�=49X=�o=�%=Y�=H�9=�7L=�P=#�
=<j=��=�7L=H�9=P�`=L��=���=��T=�t�=�
==��P=���=�F>E��B	��B 4VB#j�BCjB"�7B�2B�<B7�B�eB�@B��B��BW�BǛB*B: B��B�B�B!bmB��B��B.B��B:�B\{A��Br�B%L4B�B-�Bt�B �YB&E�B^GB�B�3B��B��B��BdkB 8?BVB*2B};B.9�B B��B)��B	8MB��B�B�B%�aB$��B�VB'�B0�B��B��A��\BwB�cB	��B >B#ABB�B"�'B�@BNB92B	%�B�+B�5B�B�7B��B?yB5B��B�9B�B!NuBy�BBF�B�&B$#BA�A�~�B�#B%BB �B�KB�B `�B&CB�B1BS�BH�B��B��B@�B��B?�B?�B�'B.r�B>�B�SB*=�B	@�B B?B�nB&-;B$�SBI�B?�B=�BĚB<MA�l#B=B��A�UA�ӎ@�A�/A5��A��Ahc�@��CA��A,xB QA��Aj��A��A�k�BȐB�=A�S|@�V?���Ale2AcxIA��Bl\@让?�6�A�zpA��A)�A0�AH�|A�;�AiY�A':�A�`A�k.A��!AB�:A�~tA<J�AA	xA��A�C�f A�zA�ZC�O�A��A MA��#A�T~A��A���@��AM@��@~y�@�2�?�j@�A��@�P�C�ͻAҋ�A���@�LA�}�A5�A�aAi�@�hA��@A9�B�A�~nAk�A���A��B��B��A�w�@���?���AlmVAc3A�oBBAC@�+?��+A��	A�%�AgA/Z�AJ�A��,Ai�JA'� A���A�*�A�{�AB��A�}EA<ɚA@c[A�c*AC�g�A�
^A�AC�H3A�vaA��A��;A�y�A�rA���@�+xAL�@�8�@|�@��\?��X@��A�nd@��C�Ծ                              	      b                                       0   ,   #         '      )               
      .   /         %   "         $      	                           	   &         -   o   %         %      #                  G                     !                  #      !         #                           !                  #   !                                                %                                                                              !      !                                                      #   !                                                %   O�ݘO:�N��dNY�#ND�)O~*O]¶N��O���N5P�O�Or�O+?%M��N	vNύ�O`��N.�O&N�BbN�O�N��NfnmN2��O���N偁O��OVfN@|O�VDO'tO��GN��DN�)%N+KGN
:�N第OgFN�@�O�U�NTN��O��/N�F%O��O�lON�aN�l:OBEVN�A�O2Y�N��Nָ�NJ�O?b~N�ͭN�\Ozk�N\�PN�ԅO�yVO56  �  [      "  ]  �  I    �  *  �  b  {  �  �  k  �  v  a  u  �  `  l  �  g  �  q    �  E  �  v  9  5  �  �  B  m  �  a  f  d  �  �  ~  k  ?    Y  �  �  �  E  4  n  �  �  S  �  o  �  ����
�o�o<t�;�o<49X;�o;�o;��
;ě�<#�
=u<#�
<o<t�<49X<D��<49X<e`B<�o<D��<���<u<T��<���<ě�<�t�<�9X<�C�<�9X<�C�<�`B<���<���<�1<�9X<�9X<�9X=,1<��<���<��=C�<�<��<��=C�=o=+=C�=+='�=#�
='�=49X=T��=y�#=�o=��=�C�=�t�=���=�S�INdgt}��������tgXPNIkhkmprz����������zmk������  ����������klnvz�����znkkkkkkkk�����������������������������������������������"'"%��dht����tohdddddddddd=<>CN[gtv��ytg[ZOB=OMB?BFOY[[_[[OOOOOOO&#")5;BJNRTUTNDB54)& #/<HU\aff^UH</, ���
#+/3;0+(#
����������������������� ������������������������������;;>ABDN[gqsspjg^[NB;VRZ[agnjg[VVVVVVVVVV��������������������������������������������%%�������HCENNO[fgigb[NHHHHHHKCOQ[fhkojh[VOKKKKKK�������������������� �)6BEKNJ@6) a_chlt��������utphaa"/>F[a_VJ;/")&!)5?BLNXVNIB54)))"#*0110$#""""""""JLOV[`ht�������th[OJICCJO[]`ehkmomh^[ROI����������������������������������������/,+-0<BHIMMKI<70////����������������������	����������)*557965){v�����������������{"###./<DHPUUNHB</-#"��������������������)5651)��������������������'+5BN[ceec_[TNB?50*'�}{������������������
)5BQXVMB4)��������*5850*�����#)/<<?AB?<3/##)6BOZVOB6)��������������������NPR[gkt}�����ujg[WRN����������GGFHMUaknwxxtnja\UJG����


!
�������! #$0<@IKJID<50(#!!		
#&$#
								����������-,6BOPWOLB76--------���������������������������������������][amnvz~��~zmmheca]]svqz��������������}s��������

 �����������)�5�)�����������������������������������������������������������������������ûлػֻлû����������������������������������������������������������������(�*�4�7�4�0�(�������&�(�(�(�(�(�(��*�;�C�N�H�?�6�*�����������������.�;�G�`�m�y�����������y�m�`�T�G�;�(�"�.�������������������������������������������)�5�F�I�E�8�5������������������l�l�n�y���������������z�y�l�l�l�l�l�l�l�h�uƁƎƗƚƝƚƘƎƁ�u�h�b�\�[�[�\�d�h������ �'�*�'�!���������������m�y�~�������������y�m�`�T�C�G�H�P�T�a�m�(�5�A�N�N�N�B�A�;�5�(�(�(�(�(�(�(�(�(�(ù������������ùõòùùùùùùùùùù�����������������������������������������$�'�0�;�=�2�$�������������������������������������������������������������ûлܻ����	������ܻԻлȻû������'�3�4�8�9�8�3�,�'������'�'�'�'�'�'�m�y�z�����������y�m�j�h�d�i�m�m�m�m�m�m�.�;�G�T�T�T�Q�G�E�;�7�.�&�(�.�.�.�.�.�.�<�H�R�U�]�a�c�a�U�H�E�=�<�:�<�<�<�<�<�<ƳƸƽƾƵƳƱƧƧƣƧƯƳƳƳƳƳƳƳƳ���������������������r�f�Y�P�K�R�f�r�����@�L�Y�a�e�l�j�e�\�Y�L�I�@�8�3�3�3�4�@�@�"�/�;�T�a�d�m�o�a�T�H�;�/�"������"ĦĳĿ����������ĿĽĳĦĞĚĖĘĚĦĦĦ�ּ������������ּּּּּּּּּֽ�����#�(�4�0�&������ݽӽʽȽֽ�����������ľ¾�������������s�g�s�t������������(�,�.�)��������տҿԿݿ�T�`�c�m�y�{���y�m�m�`�T�O�K�N�S�T�T�T�T���Ľнݽ��ݽнϽĽ��������������������������	�	���	���������������������¿½¿�����������������������t¦²µ²§¦�t�s�m�p�t�t�Z�f�w�����������������s�f�Z�V�R�N�W�Z���������������������������������(�4�A�M�Z�`�f�s�{�u�f�Z�M�A�4�(���!�(�f�m�n�o�m�f�c�Z�V�U�Z�]�f�f�f�f�f�f�f�f�/�<�A�H�O�H�G�?�<�8�/�+�&�#�!�#�%�/�/�/�����������ݿѿȿĿ��¿ĿͿݿ��E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��#�<�b�{ŅŉŇŀ�{�n�b�U�N�I�=�6�!���#�m�y���������������y�l�`�S�M�G�I�L�S�Z�mD�D�D�EEEEE%EEEED�D�D�D�D�D�D�D��m�z�~�������������z�v�s�u�s�m�j�m�m�m�m������������������������������������������������������������s�g�f�Z�N�J�O�Z�s���/�;�H�H�T�V�U�T�H�;�/�"�"��"�*�/�/�/�/�����������������������������������������)�5�B�B�N�N�N�K�B�5�)�"��!�)�)�)�)�)�)�f�r���������������r�n�f�_�Y�V�Y�`�f�f�������ľǾ������������������������������:�F�S�_�g�l�w�x�x�l�_�S�F�9�-�!���-�:�:�F�O�P�G�F�:�/�-�)�-�.�:�:�:�:�:�:�:�:�����%������������������'�3�C�L�R�L�@�3�'�������������'�r�s�~����������������~�r�q�h�h�r�r�r�r������������������ŹŭťŬŭŹ���������߼����ּݼ������ּʼ���������������D�D�D�D�D�D�D�D�D�D�D�D�D�D{DzDxD{D{D�D� 2 $ 2 ) 4 C a U $ J  0 : ^ W N ( $ { G B  a } $  + C 8 " ^  2 F } } O R  4 ^ Y %  i \ 3 � a 1 8 5  * 8 9 ; = ; } p Z     �  �  �  m  _    �  K  ?  Y  W  �  �  ,  B  �  �  +    �  �  �  �  �  �  �  �  .  C  !  X  5  �  �  �  �  #  �      �  �    	  �  �  @  ?  �  �  �  �  �  �  `  �  �  ?    �  #  @  }  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  k  t  �  �  �  �  |  j  Q  4    �  �  �  a    �  �  %   v  @  D  I  P  W  [  W  L  <  4  +    �  �  �  j    �  0   �  �  �                  �  �  �    G    �  �  J    �  �  �  �  �  �  �  �          �  �  �  �  �  h  H  %         !          �  �  �  �  �  �  w  c  O  <  (      /  G  T  [  \  [  X  O  D  4      �  �  �  =    �  !  �  �  �  �  �  �  �  �  �  �  �  x  u  r  j  [  L  1     �  I  @  7  -  $        �  �  �  �  �  �  �  �  �  �  �  �        �  �  �  �  �  �  �  g  J  /    �  �  �  �  a  �  �  �  �  �  y  m  a  T  C  0      �  �  �  �  x  W  5    �  �  &  )  )  %        �  �  �  �  x  >  �  �  {  H    �  �  �  h  �  a  �    W  u  �  }  [  #  �  R  �  �  �  �  $  >  T  _  ]  O  9  !    �  �  �  �  �  z  R  *  �  �  (  {  g  S  ?  +    �  �  �  �  �  y  h  X  H  ?  9  3  ,  &  �  �  �  �  �  �  f  <    �  �  >  �  �  ^    �  {  ,   �  �  �  �  �  �  �  �  �  �  �  r  W  :    �  �  �  �  ~  T  U  h  h  d  _  X  P  J  A  3       �  �  �  �  O    �  "  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  \  s  b  v  E    �    �  �  [    �    �  �  S  !  �  �      $  .  *  .  P  ^  a  ]  W  M  6    �  �  �  �  C  �  u  o  i  c  [  S  K  A  7  -  "    	  �  �  �  �  �  �  �  �           1  i  �  �  r  ^  F  +    �  �  s  $  �  �  �  �    =  V  i  v  v  w  x  z  w  u  t  �  �  �  �  �  �  l  `  T  H  ;  -         �  �  �  �  �  �  p  V  :      w  �  �  �  �  �  �  �    \  0  �  �  |  -  �  g  �  �  �  �    7  O  `  g  d  S  C  ,  
  �  �  M  �  �  2  �  �  �  �  �  �  �  �  �  n  F  $    �  �    \  F    �    �  �  �  "  f  l  q  c  L  +  �  �  �  @  �  o  �  w     �  s    �  �  �       �  �  �  �  �  �  �  �  �  �  x  d  P  <  )  �  �  �  �  �  �  y  J    �  �  f  *  �  �  v  B    �  �  E  >  5  ,         �  �  �  �  v  D    �  �  }  K     �  L  d  s  ~  �  �  u  \  :    �  �  d     �  R  �  8  �  �  v  u  t  s  r  q  p  i  `  W  N  E  <  0          �   �   �  �       /  5  8  9  4  .    �  �  �  <  �  |    �  �  <  5  ,  #        �  �  �  �  �  �  �  �  �  z  h  V  C  1  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  v  o  h  a  �  �  m  T  ;  (      �  �  �  �  �  �  }  m  l  i  X  F  B  @  ?  ?  ?  @  @  <  0      �  �  �  �  �  \  1      �  o  �  =  �  �    ?  X  j  j  X  +  �  �  j  �  T  g  4   �  �  �  �  �  �  �  �  �  �  �  ]     �  �  U  �  {  �  )  T  a  ]  Z  V  R  N  G  @  9  2  /  .  -  ,  ,  0  5  ;  A  F  �  0  P  a  f  b  T  >  "  �  �  �  �  �  �  �  �  ~  f    4  P  `  d  X  @  "     �  �  �  M  	  �  P  �  o  �  X  �  �  �  �  �  �  g  O  -    �  �  J  �  v  �  ^  �  �    8  �  �  �  �  �  m  D    �  �      �  �  �  ]  %  �  �  i  ~  _  D  %        �  �  �  �  �  �  �  X    �  Y   �   �  i  k  f  ]  O  9    �  �  �  Y    �  �  4  �  b  �    7  ?  <  8  5  2  ,  &  !      
    �  �  �  �  x  [  >  !  {  |  ~  }  z  v  q  l  `  R  @  +      �  �  �  �  �  �  M  U  X  U  G  8  %    �  �  �  �  f  .  �  �  �  L   �   c  �  �  �  �  �  �  �  �  �  }  t  k  `  V  K  @  6  +  !    �  �  �  �  �  �  �  �  �  �  ]  :    �  �     �  ?  �  5  �  �  �  �  �  �  �  h  N  1    �  �  �  �  y  [  =  �  �  E  A  <  2  %    �  �  �  �  �  ~  c  I  0      �    2  4  (        �  �  �  �  �  �  �  �  �  l  T  ;  !     �  n  i  _  R  ?  $  �  �  �  y  L  !  �  �  �  �  �  _  �  �  $  ]  �  �  �  �  �  �  �  �  �  _  5    �  �    H  �  �  �  �  �  �  �  �  t  b  Q  @  .    
  �  �  �  �  �  �  �  A  9  I  H  )    �  �  Z    �    8  �  �  ,  �  �  2  �  �  �  �  ~  g  Q  :    �  �  �  �  �  a  <    �  �  �  �  o  R  4    �  �       �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  a  1  �  �  v  *  �  �    g  �  �  �    C  r  �  �  �  �  p  	  �    k  �    B  (  �  
�  ^  �